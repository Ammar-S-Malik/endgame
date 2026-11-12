#!/usr/bin/env python3
"""GPU trainer for Endgame's NNUE ((768 x 8 king buckets) -> H x2 -> 1 of 8 output heads, clipped ReLU).

This is a separate tool: the engine itself has no dependencies, this script needs
PyTorch and NumPy (see requirements.txt). It reads the text files written by
`datagen` and writes a network file the engine loads with `EvalFile` or embeds from
`net/`.

    python trainer/train.py --data "data.*.txt" --out net.nnue --hidden 256

The model, loss, optimiser and file format are the same as the CPU trainer in
`src/bin/trainer.rs`:

    target = lambda * sigmoid(search score / 400) + (1 - lambda) * game result
    loss   = mean squared error between sigmoid(network output) and target

everything from the side to move's point of view, Adam with cosine learning-rate decay.

The first run converts each text file to a compact binary cache next to it
(`<file>.bin`, 68 bytes per position), one process per file. Later runs reuse the cache.
"""

import argparse
import glob
import math
import multiprocessing
import os
import struct
import sys
import time

import numpy as np

# Must match src/nnue.rs.
BUCKET_INPUTS = 768  # features in one king bucket
KING_BUCKETS = 8
INPUTS = BUCKET_INPUTS * KING_BUCKETS
OUT_BUCKETS = 8
QA = 255
QB = 64
SCALE = 400
MAGIC = b"ENN2"
OLD_MAGIC = b"ENNU"  # the previous format: 768 inputs, one output head
HEADER = 16

PAD_CODE = 12  # "no piece" in a position record
PAD_FEATURE = INPUTS  # extra all-zero feature row used for padding
MAX_PIECES = 32
PIECES = "PNBRQKpnbrqk"

# One position: up to 32 (piece code, square) pairs, padded with PAD_CODE.
# score: search score in centipawns, White's point of view.
# result2: game result for White times two (0, 1 or 2).  stm: 0 = White to move.
RECORD = np.dtype(
    [("code", "u1", MAX_PIECES), ("sq", "u1", MAX_PIECES), ("score", "<i2"), ("result2", "u1"), ("stm", "u1")]
)
assert RECORD.itemsize == 68


# ---------------------------------------------------------------------------
# Data: text -> binary cache -> NumPy
# ---------------------------------------------------------------------------

def parse_line(line):
    """Parse 'fen;score;result' into packed record bytes, or None if malformed."""
    parts = line.split(";")
    if len(parts) < 3:
        return None
    try:
        score = float(parts[1])
        result = float(parts[2])
    except ValueError:
        return None
    fields = parts[0].split(" ")
    if len(fields) < 2:
        return None
    codes = bytearray()
    squares = bytearray()
    rank, file = 7, 0
    for c in fields[0]:
        if c == "/":
            rank -= 1
            file = 0
        elif "1" <= c <= "8":
            file += ord(c) - 48
        else:
            code = PIECES.find(c)
            sq = rank * 8 + file
            if code < 0 or not 0 <= sq < 64:
                return None
            codes.append(code)
            squares.append(sq)
            file += 1
    n = len(codes)
    if n == 0 or n > MAX_PIECES:
        return None
    pad = MAX_PIECES - n
    score_i = max(-32767, min(32767, int(round(score))))
    result2 = int(round(result * 2))
    if result2 not in (0, 1, 2):
        return None
    stm = 0 if fields[1] == "w" else 1
    return (
        bytes(codes) + bytes([PAD_CODE]) * pad + bytes(squares) + bytes(pad) + struct.pack("<hBB", score_i, result2, stm)
    )


def cache_path(path):
    return path + ".bin"


def build_cache(path):
    """Convert one text file to its binary cache. Returns (path, positions, skipped)."""
    out = cache_path(path)
    tmp = out + ".tmp"
    count = skipped = 0
    with open(path, "r", encoding="ascii", errors="replace") as src, open(tmp, "wb") as dst:
        buf = []
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = parse_line(line)
            if rec is None:
                skipped += 1
                continue
            buf.append(rec)
            count += 1
            if len(buf) >= 65536:
                dst.write(b"".join(buf))
                buf.clear()
        dst.write(b"".join(buf))
    os.replace(tmp, out)
    return path, count, skipped


def cache_is_fresh(path):
    out = cache_path(path)
    return (
        os.path.exists(out)
        and os.path.getmtime(out) >= os.path.getmtime(path)
        and os.path.getsize(out) % RECORD.itemsize == 0
    )


def load_data(paths, workers):
    todo = [p for p in paths if not cache_is_fresh(p)]
    if todo:
        print(f"converting {len(todo)} text file(s) to binary cache...", flush=True)
        t0 = time.time()
        if len(todo) > 1 and workers > 1:
            with multiprocessing.Pool(min(workers, len(todo))) as pool:
                results = pool.map(build_cache, todo)
        else:
            results = [build_cache(p) for p in todo]
        for path, count, skipped in results:
            note = f", {skipped} malformed lines skipped" if skipped else ""
            print(f"  {path}: {count} positions{note}")
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
    parts = [np.fromfile(cache_path(p), dtype=RECORD) for p in paths]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


# ---------------------------------------------------------------------------
# Network file (same layout as Net::to_bytes in src/nnue.rs)
# ---------------------------------------------------------------------------

def round_half_away(x):
    """Rust's f32::round: halves round away from zero (NumPy rounds them to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantise(params):
    """Float parameters -> the engine's integers.

    params: ft_w [INPUTS,H], ft_b [H], out_w [OUT_BUCKETS,2H], out_b [OUT_BUCKETS].
    """
    q16 = lambda v, sc: np.clip(round_half_away(np.asarray(v, dtype=np.float32) * np.float32(sc)), -32767, 32767).astype("<i2")
    out_b = round_half_away(np.asarray(params["out_b"], dtype=np.float32) * np.float32(QA * QB))
    return {
        "ft_w": q16(params["ft_w"], QA),
        "ft_b": q16(params["ft_b"], QA),
        "out_w": q16(params["out_w"], QB),
        "out_b": np.clip(out_b, -(2**31), 2**31 - 1).astype("<i4"),
    }


def net_to_bytes(q):
    h = q["ft_b"].shape[0]
    assert q["ft_w"].shape == (INPUTS, h)
    assert q["out_w"].shape == (OUT_BUCKETS, 2 * h) and q["out_b"].shape == (OUT_BUCKETS,)
    return (
        MAGIC
        + struct.pack("<III", h, KING_BUCKETS, OUT_BUCKETS)
        + q["ft_w"].astype("<i2").tobytes()
        + q["ft_b"].astype("<i2").tobytes()
        + q["out_w"].astype("<i2").tobytes()
        + q["out_b"].astype("<i4").tobytes()
    )


def net_from_bytes(data):
    """Read a v2 network file."""
    if len(data) < HEADER or data[:4] != MAGIC:
        raise ValueError("not an Endgame v2 NNUE file")
    h, kb, ob = struct.unpack("<III", data[4:HEADER])
    if kb != KING_BUCKETS or ob != OUT_BUCKETS:
        raise ValueError(f"network has {kb} king buckets and {ob} output buckets, expected {KING_BUCKETS} and {OUT_BUCKETS}")
    expected = HEADER + 2 * (INPUTS * h + h + OUT_BUCKETS * 2 * h) + 4 * OUT_BUCKETS
    if len(data) != expected:
        raise ValueError(f"bad file size {len(data)} (expected {expected} for {h} hidden units)")
    pos = HEADER
    ft_w = np.frombuffer(data, "<i2", INPUTS * h, pos).reshape(INPUTS, h)
    pos += 2 * INPUTS * h
    ft_b = np.frombuffer(data, "<i2", h, pos)
    pos += 2 * h
    out_w = np.frombuffer(data, "<i2", OUT_BUCKETS * 2 * h, pos).reshape(OUT_BUCKETS, 2 * h)
    pos += 2 * OUT_BUCKETS * 2 * h
    out_b = np.frombuffer(data, "<i4", OUT_BUCKETS, pos)
    return {"ft_w": ft_w.copy(), "ft_b": ft_b.copy(), "out_w": out_w.copy(), "out_b": out_b.copy()}


def old_net_from_bytes(data):
    """Read a v1 network file (768 inputs, a single output head)."""
    if len(data) < 8 or data[:4] != OLD_MAGIC:
        raise ValueError("not an old Endgame NNUE file")
    h = struct.unpack("<I", data[4:8])[0]
    expected = 8 + 2 * (BUCKET_INPUTS * h + h + 2 * h) + 4
    if len(data) != expected:
        raise ValueError(f"bad file size {len(data)} (expected {expected} for {h} hidden units)")
    pos = 8
    ft_w = np.frombuffer(data, "<i2", BUCKET_INPUTS * h, pos).reshape(BUCKET_INPUTS, h)
    pos += 2 * BUCKET_INPUTS * h
    ft_b = np.frombuffer(data, "<i2", h, pos)
    pos += 2 * h
    out_w = np.frombuffer(data, "<i2", 2 * h, pos)
    pos += 4 * h
    out_b = struct.unpack("<i", data[pos : pos + 4])[0]
    return {"ft_w": ft_w.copy(), "ft_b": ft_b.copy(), "out_w": out_w.copy(), "out_b": out_b}


def expand_old(old):
    """Turn a v1 network into a v2 one that evaluates every position identically.

    Every king bucket gets a copy of the v1 feature weights and every output head a copy of
    the v1 output weights, so training can then specialise them.
    """
    return {
        "ft_w": np.tile(old["ft_w"], (KING_BUCKETS, 1)),
        "ft_b": old["ft_b"].copy(),
        "out_w": np.tile(old["out_w"][None, :], (OUT_BUCKETS, 1)),
        "out_b": np.full(OUT_BUCKETS, old["out_b"], dtype="<i4"),
    }


def net_from_any_bytes(data):
    """Read a v2 network, or a v1 network expanded to v2."""
    if data[:4] == OLD_MAGIC:
        return expand_old(old_net_from_bytes(data))
    return net_from_bytes(data)


def dequantise(q):
    return {
        "ft_w": q["ft_w"].astype(np.float32) / QA,
        "ft_b": q["ft_b"].astype(np.float32) / QA,
        "out_w": q["out_w"].astype(np.float32) / QB,
        "out_b": q["out_b"].astype(np.float32) / (QA * QB),
    }


# ---------------------------------------------------------------------------
# Features and the integer network in NumPy (used to check the exported file)
# ---------------------------------------------------------------------------

def king_bucket_np(perspective, ksq):
    """King bucket (0..7) for a perspective whose own king stands on real-board square ksq.

    Same mapping as nnue::king_bucket: the square is seen from the perspective's side, rank
    groups {0}, {1}, {2,3}, {4..7} times file halves {a-d}, {e-h}.
    """
    rel = ksq if perspective == 0 else ksq ^ 56
    rank = rel // 8
    file = rel % 8
    rank_group = (rank > 0).astype(np.int64) + (rank > 1) + (rank > 3)
    return rank_group * 2 + (file >= 4)


def king_squares_np(code, sq):
    """(white king square, black king square) of every position."""
    code = code.astype(np.int64)
    sq = sq.astype(np.int64)
    wk = np.take_along_axis(sq, (code == 5).argmax(1)[:, None], 1)[:, 0]
    bk = np.take_along_axis(sq, (code == 11).argmax(1)[:, None], 1)[:, 0]
    return wk, bk


def output_bucket_np(code):
    """Output head (0..7) from the number of pieces on the board, kings included."""
    n = (code != PAD_CODE).sum(1).astype(np.int64)
    return np.minimum(np.maximum(n - 2, 0) // 4, OUT_BUCKETS - 1)


def features_np(code, sq, stm):
    """Feature indices [N,32] for the side to move ("us") and the other side ("them").

    Same mapping as nnue::feature: White's perspective is bucket*768 + code*64 + sq; Black's
    swaps the piece colours and mirrors the board vertically. The bucket comes from that
    perspective's own king. Padding maps to PAD_FEATURE.
    """
    wk, bk = king_squares_np(code, sq)
    code = code.astype(np.int64)
    sq = sq.astype(np.int64)
    pad = code == PAD_CODE
    wb = king_bucket_np(0, wk)[:, None] * BUCKET_INPUTS
    bb = king_bucket_np(1, bk)[:, None] * BUCKET_INPUTS
    white = np.where(pad, PAD_FEATURE, wb + code * 64 + sq)
    black = np.where(pad, PAD_FEATURE, bb + ((code + 6) % 12) * 64 + (sq ^ 56))
    white_to_move = (stm == 0)[:, None]
    return np.where(white_to_move, white, black), np.where(white_to_move, black, white)


def targets_np(rec, lam):
    s = rec["score"].astype(np.float32)
    r = rec["result2"].astype(np.float32) / 2
    black = rec["stm"] == 1
    s = np.where(black, -s, s)
    r = np.where(black, 1 - r, r)
    return lam / (1 + np.exp(-s / SCALE)) + (1 - lam) * r


def integer_eval_np(q, rec):
    """Centipawn evaluation exactly as the engine computes it. Returns (cp, largest |accumulator|)."""
    h = q["ft_b"].shape[0]
    table = np.vstack([q["ft_w"].astype(np.int64), np.zeros((1, h), np.int64)])
    us, them = features_np(rec["code"], rec["sq"], rec["stm"])
    acc_us = table[us].sum(1) + q["ft_b"].astype(np.int64)
    acc_them = table[them].sum(1) + q["ft_b"].astype(np.int64)
    peak = int(max(np.abs(acc_us).max(), np.abs(acc_them).max()))
    bucket = output_bucket_np(rec["code"])
    w = q["out_w"].astype(np.int64)[bucket]  # [N, 2H]
    total = (
        q["out_b"].astype(np.int64)[bucket]
        + (np.clip(acc_us, 0, QA) * w[:, :h]).sum(1)
        + (np.clip(acc_them, 0, QA) * w[:, h:]).sum(1)
    )
    total = total * SCALE
    cp = np.sign(total) * (np.abs(total) // (QA * QB))  # Rust integer division truncates toward zero
    return cp, peak


def quantised_loss_np(q, rec, lam, chunk=8192):
    total, peak = 0.0, 0
    for i in range(0, len(rec), chunk):
        part = rec[i : i + chunk]
        cp, p = integer_eval_np(q, part)
        pred = 1 / (1 + np.exp(-cp.astype(np.float64) / SCALE))
        total += float(((pred - targets_np(part, lam)) ** 2).sum())
        peak = max(peak, p)
    return total / max(len(rec), 1), peak


# ---------------------------------------------------------------------------
# Training (PyTorch)
# ---------------------------------------------------------------------------

def train(args, data):
    try:
        import torch
        from torch import nn
    except ImportError:
        sys.exit("PyTorch is not installed. See trainer/requirements.txt for the install command.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        print(f"training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"training on {device} (no CUDA GPU found or selected, this will be slow)")
    torch.manual_seed(args.seed)
    h = args.hidden
    lam = args.lambda_

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Row PAD_FEATURE is the padding row: always zero, never trained.
            self.ft = nn.EmbeddingBag(INPUTS + 1, h, mode="sum", padding_idx=PAD_FEATURE)
            self.ft_b = nn.Parameter(torch.zeros(h))
            # One output head per piece-count bucket; weight [OUT_BUCKETS, 2h], bias [OUT_BUCKETS].
            self.out = nn.Linear(2 * h, OUT_BUCKETS)

        def forward(self, us, them, ob):
            a = torch.clamp(self.ft(us) + self.ft_b, 0.0, 1.0)
            b = torch.clamp(self.ft(them) + self.ft_b, 0.0, 1.0)
            heads = self.out(torch.cat([a, b], dim=1))  # [N, OUT_BUCKETS]
            return heads.gather(1, ob.unsqueeze(1)).squeeze(1)

    model = Model().to(device)
    with torch.no_grad():
        if args.init:
            with open(args.init, "rb") as f:
                p = dequantise(net_from_any_bytes(f.read()))
            if p["ft_b"].shape[0] != h:
                sys.exit(f"--init network has {p['ft_b'].shape[0]} hidden units, --hidden is {h}")
            model.ft.weight[:INPUTS] = torch.from_numpy(p["ft_w"]).to(device)
            model.ft_b.copy_(torch.from_numpy(p["ft_b"]).to(device))
            model.out.weight.copy_(torch.from_numpy(p["out_w"]).to(device))
            model.out.bias.copy_(torch.from_numpy(p["out_b"]).to(device))
            print(f"initialised from {args.init}")
        else:
            model.ft.weight.uniform_(-0.1, 0.1)
            model.out.weight.uniform_(-0.1, 0.1)
            model.out.bias.zero_()
        model.ft.weight[PAD_FEATURE].zero_()

    def to_batch(rec):
        """Position records -> (us features, them features, target) on the device."""
        code = torch.from_numpy(np.ascontiguousarray(rec["code"])).to(device).long()
        sq = torch.from_numpy(np.ascontiguousarray(rec["sq"])).to(device).long()
        stm = torch.from_numpy(np.ascontiguousarray(rec["stm"])).to(device)
        score = torch.from_numpy(np.ascontiguousarray(rec["score"])).to(device).float()
        result = torch.from_numpy(np.ascontiguousarray(rec["result2"])).to(device).float() / 2
        pad = code == PAD_CODE
        # Own king squares and the king buckets they select (see nnue::king_bucket).
        wk = sq.gather(1, (code == 5).long().argmax(1, keepdim=True)).squeeze(1)
        bk = sq.gather(1, (code == 11).long().argmax(1, keepdim=True)).squeeze(1)

        def bucket_of(rel):
            rank = rel // 8
            rank_group = (rank > 0).long() + (rank > 1).long() + (rank > 3).long()
            return rank_group * 2 + ((rel % 8) >= 4).long()

        wb = (bucket_of(wk) * BUCKET_INPUTS).unsqueeze(1)
        bb = (bucket_of(bk ^ 56) * BUCKET_INPUTS).unsqueeze(1)
        white = (wb + code * 64 + sq).masked_fill(pad, PAD_FEATURE)
        black = (bb + ((code + 6) % 12) * 64 + (sq ^ 56)).masked_fill(pad, PAD_FEATURE)
        pieces = (~pad).sum(1)
        ob = (((pieces - 2).clamp(min=0)) // 4).clamp(max=OUT_BUCKETS - 1)
        wtm = (stm == 0).unsqueeze(1)
        us = torch.where(wtm, white, black)
        them = torch.where(wtm, black, white)
        black_to_move = stm == 1
        score = torch.where(black_to_move, -score, score)
        result = torch.where(black_to_move, 1 - result, result)
        target = lam * torch.sigmoid(score / SCALE) + (1 - lam) * result
        return us, them, ob, target

    n = len(data)
    val_n = min(max(n // 50, 1), 100_000)
    train_n = n - val_n
    val = data[train_n:]
    print(f"{n} positions ({train_n} train, {val_n} validation)")
    if train_n < args.batch:
        sys.exit("not enough data for one batch")

    def val_loss():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, val_n, 65536):
                us, them, ob, target = to_batch(val[i : i + 65536])
                total += float(((torch.sigmoid(model(us, them, ob)) - target) ** 2).sum())
        model.train()
        return total / val_n

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(args.seed)
    order = np.arange(train_n, dtype=np.uint32)
    print(f"initial validation loss {val_loss():.5f}", flush=True)
    t_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        rng.shuffle(order)
        lr = args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs)))
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        seen = 0
        for i in range(0, train_n, args.batch):
            idx = order[i : i + args.batch]
            if len(idx) < args.batch // 2:
                continue
            us, them, ob, target = to_batch(data[np.sort(idx)])
            loss = ((torch.sigmoid(model(us, them, ob)) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
            seen += len(idx)
        print(
            f"epoch {epoch + 1:>3}/{args.epochs}  lr {lr:.5f}  train loss {float(loss_sum) / max(seen, 1):.5f}"
            f"  val loss {val_loss():.5f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    print(f"training took {time.time() - t_start:.0f}s")

    params = {
        "ft_w": model.ft.weight[:INPUTS].detach().cpu().numpy(),
        "ft_b": model.ft_b.detach().cpu().numpy(),
        "out_w": model.out.weight.detach().cpu().numpy(),
        "out_b": model.out.bias.detach().cpu().numpy(),
    }
    return params, val, val_loss()


def main():
    ap = argparse.ArgumentParser(description="GPU trainer for Endgame's NNUE")
    ap.add_argument("--data", required=True, help='datagen text files: comma-separated, wildcards allowed ("data.*.txt")')
    ap.add_argument("--out", default="net.nnue")
    ap.add_argument("--hidden", type=int, default=256, help="hidden layer size; must equal H in src/nnue.rs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.75, help="weight of the search score vs. the game result")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", help="start from an existing network file (v2, or an old v1 file, which is expanded)")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes for the text-to-binary conversion")
    args = ap.parse_args()

    paths = []
    for part in args.data.split(","):
        if not part:
            continue
        # Expand wildcards here because the Windows shell does not. Never pick up our own caches.
        found = sorted(f for f in glob.glob(part) if not f.endswith((".bin", ".tmp"))) if glob.has_magic(part) else [part]
        if not found:
            sys.exit(f"no files match {part}")
        paths.extend(found)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"cannot find {', '.join(missing)}")

    t0 = time.time()
    data = load_data(paths, args.workers)
    print(f"loaded {len(data)} positions in {time.time() - t0:.1f}s", flush=True)

    params, val, float_loss = train(args, data)

    q = quantise(params)
    int_loss, peak = quantised_loss_np(q, val, args.lambda_)
    print(f"validation loss: float {float_loss:.5f}, quantised integer network {int_loss:.5f}")
    if peak > 32767:
        print(f"WARNING: accumulator reached {peak}, which overflows the engine's 16-bit accumulators.")
        print("         Do not use this network; retrain with a lower --lr or fewer --epochs.")
    blob = net_to_bytes(q)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {args.out} ({len(blob)} bytes)")
    print(f"the engine must be built with H = {args.hidden} in src/nnue.rs to load it")


if __name__ == "__main__":
    main()
