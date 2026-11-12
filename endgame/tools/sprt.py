#!/usr/bin/env python3
"""SPRT match runner: decides whether a NEW engine build is stronger than an OLD one.

Plays paired games (each random opening is played with both colors) over UCI and stops
as soon as the sequential probability ratio test accepts H1 (NEW is at least `elo1`
stronger) or H0 (NEW is at most `elo0` stronger), or at --max-games.

usage: sprt.py NEW_BIN OLD_BIN [--movetime 40] [--workers 2] [--elo0 0] [--elo1 10]
                               [--max-games 2000] [--hash 16] [--seed 1] [--log FILE]

NEW_BIN must support the non-standard `status` and `genopening` commands (all builds
from this repo do). Exit code: 0 = H1 accepted, 1 = H0 accepted, 2 = inconclusive.
"""
import argparse
import concurrent.futures as cf
import math
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("new")
ap.add_argument("old")
ap.add_argument("--movetime", type=int, default=40, help="ms per move")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--elo0", type=float, default=0.0)
ap.add_argument("--elo1", type=float, default=10.0)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--beta", type=float, default=0.05)
ap.add_argument("--max-games", type=int, default=2000)
ap.add_argument("--min-games", type=int, default=60, help="never stop before this many games")
ap.add_argument("--hash", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--opening-plies", type=int, default=8)
ap.add_argument("--log", default=None)
ap.add_argument("--new-opt", action="append", default=[], metavar="NAME=VALUE",
                help="UCI option to set on the NEW engine (repeatable)")
ap.add_argument("--old-opt", action="append", default=[], metavar="NAME=VALUE",
                help="UCI option to set on the OLD engine (repeatable)")
ap.add_argument("--opening-kind", choices=["normal", "endgame"], default="normal",
                help="'endgame' starts every game from a reduced-material position")
args = ap.parse_args()

MAX_PLIES = 300
WIN_CP, WIN_COUNT = 900, 4  # adjudicate when both sides agree on a big score for this many plies


def expected_score(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def llr(w, d, l, elo0, elo1):
    """Generalised SPRT log-likelihood ratio for a win/draw/loss count (normal approximation)."""
    # Smooth the counts slightly so a lucky start (zero variance) cannot produce a huge LLR.
    w, d, l = w + 0.5, d + 0.5, l + 0.5
    n = w + d + l
    mean = (w + 0.5 * d) / n
    var = (w * (1 - mean) ** 2 + d * (0.5 - mean) ** 2 + l * (0 - mean) ** 2) / n
    var = max(var, 1e-6)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


class Engine:
    def __init__(self, path, options=()):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.send("uci")
        self.wait("uciok")
        self.send(f"setoption name Hash value {args.hash}")
        for opt in options:
            name, _, value = opt.partition("=")
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.wait("readyok")
        self.last_score = 0

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(prefix):
                return line.strip()
            if line.startswith("info string") and ("could not" in line or "error" in line):
                raise RuntimeError(line.strip())

    @staticmethod
    def position(moves):
        return "position startpos" + (" moves " + " ".join(moves) if moves else "")

    def status(self, moves):
        self.send(self.position(moves))
        self.send("status")
        return self.wait("status").split()[1]

    def new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait("readyok")

    def best_move(self, moves):
        """Returns (move, score_cp_from_mover_pov)."""
        self.send(self.position(moves))
        self.send(f"go movetime {args.movetime}")
        score = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith("info ") and " score " in line:
                t = line.split()
                k = t.index("score")
                if t[k + 1] == "cp":
                    score = int(t[k + 2])
                elif t[k + 1] == "mate":
                    score = 30000 if int(t[k + 2]) > 0 else -30000
            elif line.startswith("info string") and ("could not" in line or "error" in line):
                raise RuntimeError(line.strip())
            elif line.startswith("bestmove"):
                return line.split()[1], score

    def opening(self, seed):
        self.send(f"genopening {args.opening_plies} {seed}" + (" endgame" if args.opening_kind == "endgame" else ""))
        return self.wait("opening").split()[1:]

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(opening, new_white):
    new, old = Engine(args.new, args.new_opt), Engine(args.old, args.old_opt)
    white, black = (new, old) if new_white else (old, new)
    moves = list(opening)
    result, reason = 0.5, "move limit"
    big = 0  # consecutive plies where the mover reports a decisive score for itself
    big_sign = 0
    try:
        for _ in range(MAX_PLIES):
            white_to_move = len(moves) % 2 == 0
            status = new.status(moves)
            if status == "checkmate":
                white_won = not white_to_move
                return (1.0 if white_won == new_white else 0.0), "checkmate", len(moves)
            if status != "ongoing":
                return 0.5, status, len(moves)
            eng = white if white_to_move else black
            mv, score = eng.best_move(moves)
            if mv == "0000":
                return 0.5, "no move", len(moves)
            moves.append(mv)
            # Win adjudication: the mover thinks it is winning by a lot for several plies.
            if score is not None and abs(score) >= WIN_CP:
                sign = (1 if score > 0 else -1) * (1 if white_to_move else -1)  # + = white ahead
                big = big + 1 if sign == big_sign else 1
                big_sign = sign
                if big >= WIN_COUNT:
                    white_won = sign > 0
                    return (1.0 if white_won == new_white else 0.0), "adjudicated", len(moves)
            else:
                big, big_sign = 0, 0
    finally:
        new.close()
        old.close()
    return result, reason, len(moves)


lock = threading.Lock()
W = D = L = 0
games_done = 0
stop_event = threading.Event()
verdict = None
t0 = time.time()
lower = math.log(args.beta / (1 - args.alpha))
upper = math.log((1 - args.beta) / args.alpha)
log_f = open(args.log, "w") if args.log else None


def emit(msg):
    print(msg, flush=True)
    if log_f:
        log_f.write(msg + "\n")
        log_f.flush()


def worker(pair_index):
    """Play one opening with both colors."""
    global W, D, L, games_done, verdict
    if stop_event.is_set():
        return
    probe = Engine(args.new)
    try:
        opening = probe.opening(args.seed * 100003 + pair_index)
    finally:
        probe.close()
    for new_white in (True, False):
        if stop_event.is_set():
            return
        res, reason, plies = play_game(opening, new_white)
        with lock:
            games_done += 1
            if res == 1.0:
                W += 1
            elif res == 0.0:
                L += 1
            else:
                D += 1
            ll = llr(W, D, L, args.elo0, args.elo1)
            if games_done % 20 == 0:
                n = W + D + L
                mean = (W + 0.5 * D) / n
                elo = 400 * math.log10(max(mean, 1e-9) / max(1 - mean, 1e-9)) if 0 < mean < 1 else float("nan")
                emit(
                    f"games {n:>4}  +{W} ={D} -{L}  score {mean*100:5.1f}%  elo {elo:+6.1f}  "
                    f"LLR {ll:+.2f} [{lower:.2f}, {upper:.2f}]  {time.time()-t0:5.0f}s"
                )
            if verdict is None and games_done >= args.min_games and ll >= upper:
                verdict = "H1"
                stop_event.set()
            elif verdict is None and games_done >= args.min_games and ll <= lower:
                verdict = "H0"
                stop_event.set()
            if games_done >= args.max_games:
                stop_event.set()


emit(f"SPRT NEW={args.new} OLD={args.old} movetime={args.movetime}ms elo0={args.elo0} elo1={args.elo1}")
with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(worker, i) for i in range((args.max_games + 1) // 2)]
    for f in futures:
        f.result()

n = W + D + L
mean = (W + 0.5 * D) / max(n, 1)
mc = min(max(mean, 1e-6), 1 - 1e-6)
elo = 400 * math.log10(mc / (1 - mc))
var = (W * (1 - mean) ** 2 + D * (0.5 - mean) ** 2 + L * mean**2) / max(n, 1)
se = math.sqrt(var / max(n, 1))
lo_p = min(max(mean - 1.96 * se, 1e-6), 1 - 1e-6)
hi_p = min(max(mean + 1.96 * se, 1e-6), 1 - 1e-6)
lo_e, hi_e = 400 * math.log10(lo_p / (1 - lo_p)), 400 * math.log10(hi_p / (1 - hi_p))
final = {"H1": "H1 ACCEPTED (NEW is stronger)", "H0": "H0 ACCEPTED (no gain / regression)", None: "INCONCLUSIVE (hit game limit)"}[verdict]
emit(f"RESULT: {final}")
emit(f"games {n}: +{W} ={D} -{L}  score {mean*100:.1f}%  Elo {elo:+.1f}  (95% CI {lo_e:+.0f} .. {hi_e:+.0f})  {time.time()-t0:.0f}s")
sys.exit({"H1": 0, "H0": 1, None: 2}[verdict])
