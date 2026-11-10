# Endgame

A UCI chess engine written from scratch in Rust, with no external dependencies.

Works with any UCI-compatible GUI (Arena, Cute Chess, Banksia, Lichess-bot, ...).

## Build and run

```sh
cargo build --release
./target/release/endgame            # speaks UCI on stdin/stdout
./target/release/endgame bench 7    # fixed-depth benchmark (nodes + nps)
./target/release/endgame perft 5    # move-generator node count
cargo test --release                # correctness tests
```

## What's implemented

- **Board:** bitboards + mailbox, FEN in/out, incremental Zobrist hashing, make/undo, null move
- **Move generation:** full legal chess (castling, en passant, promotions), verified with
  perft against six standard reference positions (up to ~4.9M nodes)
- **Search:** iterative deepening, negamax alpha-beta with principal variation search,
  quiescence search, null-move pruning, check extensions, repetition and 50-move draw detection
- **Transposition table:** Zobrist-keyed hash table (configurable via the UCI `Hash` option)
  with exact/lower/upper bounds, mate-score adjustment and best-move ordering
- **Move ordering:** MVV-LVA captures, killer moves, history heuristic
- **Evaluation:** material + piece-square tables with a tapered king (middlegame/endgame),
  updated incrementally in make/undo so evaluating a position is O(1)
- **UCI:** `position`, `go` (depth / movetime / wtime+btime+inc / infinite), `stop`, time management

## Testing

- Perft matches published node counts for startpos, Kiwipete and positions 3-6
- Random playouts check that the incremental hash always equals a from-scratch hash and that
  undoing every move restores the exact starting position
- Tactical tests: mate in one, mate score reporting, winning material, avoiding stalemate
- A 300-ply self-play game over UCI where every move is validated by the opposing engine
- Incremental evaluation is checked against a from-scratch recompute after every random move

## Results

Measured at depth 8 across six benchmark positions (`endgame bench 8`), adding the
transposition table and principal variation search cut the nodes needed from 16.0M to 6.0M and the
time from 2.9 s to 1.0 s. Incremental evaluation added about 10% more nodes per second.

In a 40-game self-play match at 100 ms per move (10 openings, colors swapped), this version beat
the original build 23 wins, 10 draws, 7 losses (70% score, about +147 Elo, 95% CI roughly
+57 to +262). This compares two versions of the same engine, so it is not an absolute rating.
