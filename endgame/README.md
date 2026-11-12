# Endgame

A UCI chess engine written from scratch in Rust, with no external dependencies, including its own
NNUE evaluation network, trained from the engine's self-play games. The network trainer is a
separate tool (`trainer/`, PyTorch on a GPU); the engine does not depend on it.

Works with any UCI-compatible GUI (Arena, Cute Chess, Banksia, Lichess-bot, ...).

## Build and run

```sh
cargo build --release
./target/release/endgame            # speaks UCI on stdin/stdout
./target/release/endgame bench 8    # fixed-depth benchmark (nodes + nps)
./target/release/endgame perft 5    # move-generator node count
cargo test --release                # correctness tests
```

UCI options: `Hash` (MB), `UseNNUE` (default true; false uses the handcrafted evaluation),
`EvalFile` (load a different network file).

## What's implemented

- **Board:** bitboards + mailbox, FEN in/out, incremental Zobrist hashing, make/undo, null move
- **Move generation:** full legal chess (castling, en passant, promotions), verified with
  perft against six standard reference positions (up to ~4.9M nodes)
- **Search:** iterative deepening, negamax alpha-beta with principal variation search,
  aspiration windows, quiescence search with static-exchange (SEE) pruning, null-move pruning,
  late move reductions, reverse futility / futility / late-move pruning, internal iterative
  reduction, check extensions, repetition and 50-move draw detection
- **Transposition table:** Zobrist-keyed hash table (configurable via the UCI `Hash` option)
  with exact/lower/upper bounds, mate-score adjustment and best-move ordering
- **Move ordering:** hash move, SEE-sorted captures (MVV-LVA), killer moves, history heuristic
- **Evaluation (NNUE):** an (768 x 8 king buckets) -> 64 (x2 perspectives) -> 1 network with
  clipped ReLU, 8 output heads chosen by piece count, and integer (i16) inference. The hidden
  layer is updated incrementally in make/undo (and rebuilt when a king changes bucket). The
  network is embedded in the binary (`net/endgame-v2.nnue`). A handcrafted evaluation (material +
  piece-square tables, tapered king) remains as a fallback.
- **UCI:** `position`, `go` (depth / movetime / wtime+btime+inc / infinite), `stop`, time management

## Training your own network

Training has two parts. Data generation is part of the engine's code and has no dependencies.
The trainer is a separate tool that runs on a GPU.

```sh
# 1. Self-play data: fixed-depth games, one "fen;score;result" line per quiet position
./target/release/datagen --games 40000 --depth 6 --threads 2 --net net/endgame-v1.nnue --out data
# 2. Train on the GPU (separate tool, see below) and export
python trainer/train.py --data "data.*.txt" --out net.nnue --hidden 64
# 3. Try it
#    setoption name EvalFile value net.nnue
```

`datagen --net` plays the games with a given network, so each round of data can come from the
previous network; without `--net` it uses the handcrafted evaluation. To ship a new network,
replace `net/endgame-v1.nnue` and rebuild.

`--hidden` must equal `H` in `src/nnue.rs` (currently 64). To move to a larger network, train it
with the new size, then change `H`, replace `net/endgame-v1.nnue` with the new file and rebuild.

### The GPU trainer (separate tool)

`trainer/train.py` is not part of the engine and is the only thing in this repository with
dependencies: Python 3, NumPy and PyTorch.

```sh
pip install numpy
# PyTorch needs a CUDA build to use an NVIDIA GPU; see trainer/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python trainer/train.py --data "data.*.txt" --out net.nnue --hidden 64 --epochs 30
```

It trains the same model with the same loss as the CPU trainer (Adam, mean squared error against
a blend of search score and game result) and writes the same file format. The first run converts
each text file to a binary cache next to it (`<file>.bin`); later runs reuse it. After training it
evaluates the exported integer network exactly as the engine does and reports its loss next to the
float model's. It prints which device it is using; without a CUDA GPU it falls back to the CPU.

### The CPU trainer (no dependencies)

The original trainer is still in the repository for machines without a GPU:

```sh
./target/release/trainer --data data.0.txt,data.1.txt --out net.nnue --epochs 30
```

## Testing

- Perft matches published node counts for startpos, Kiwipete and positions 3-6
- Random playouts check that the incremental hash and the incremental evaluation always equal a
  from-scratch recompute, and that undoing every move restores the exact starting position
- NNUE accumulators are checked against a full refresh after every move and undo in random games,
  and the embedded network and file format are validated
- Tactical tests: mate in one, mate score reporting, winning material, avoiding stalemate
- A self-play game over UCI where every move is validated by the opposing engine

### Strength testing

`tools/sprt.py` plays two builds against each other over UCI on random balanced openings (each
played with both colors) and stops with a sequential probability ratio test (SPRT) once the result
is statistically settled. Every search change and the network were accepted only after passing it.

```sh
python3 tools/sprt.py NEW_BIN OLD_BIN --movetime 40 --elo0 0 --elo1 15
python3 tools/sprt.py ./endgame ./endgame --old-opt UseNNUE=false --opening-kind endgame
```

## Results

All results are self-play matches at 40 ms per move on a 2-core machine (so they compare versions
of this engine with each other and are not an absolute rating):

| Change | Elo vs. previous version | Games |
|---|---|---|
| Incremental eval + transposition table + PVS | about +147 | 40 |
| Late move reductions | about +123 (95% CI +74 to +178) | 147 |
| Reverse futility, futility and late-move pruning | about +53 (+24 to +84) | 381 |
| SEE, internal iterative reduction, aspiration windows | about +82 (+43 to +122) | 212 |
| NNUE vs. handcrafted evaluation, normal openings | about +41 (+15 to +67) | 592 |
| NNUE vs. handcrafted evaluation, endgame starts | about +58 (+22 to +94) | 213 |

Overall, the current build beats the version from before these changes (transposition table,
incremental eval, no reductions or pruning, handcrafted evaluation) by about +182 Elo
(+143 to +225, 300 games). Elo gains do not add up exactly, which is why this is below the sum.

The network was trained on 3.28 million positions from 40,000 self-play games at depth 6
(about 45 minutes of generation, 92 seconds of training). More data, a larger network and
repeated rounds of self-play training are the obvious ways to make it stronger.

## Network format v2, search options and builds

- Network files are now format v2 (magic `ENN2`). Old v1 files (`ENNU`) are converted with
  `python tools/convert_net.py old.nnue new.nnue`, which keeps the evaluation identical;
  `trainer/train.py --init` also accepts a v1 file. `datagen --net` and `EvalFile` need v2 files.
- The CPU trainer (`src/bin/trainer.rs`) only supports the v1 layout and was moved to
  `tools/legacy/trainer.rs.old`; use `trainer/train.py`.
- Search features switchable through UCI options, all on by default (each passed SPRT): `Improving`, `ContHist`
  (counter-move + continuation history) and `Singular` (singular extensions). Test against the
  baseline with the same binary, e.g. `python tools/sprt.py ENGINE ENGINE --old-opt Singular=false`.
- `.cargo/config.toml` builds for the local CPU (`target-cpu=native`); delete it for a portable binary.
