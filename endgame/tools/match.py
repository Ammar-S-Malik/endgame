#!/usr/bin/env python3
"""Play NEW vs OLD Kestrel builds over UCI and report score and Elo difference.

usage: match.py NEW_BIN OLD_BIN [games=40] [movetime_ms=100] [workers=2]
"""
import concurrent.futures as cf
import math
import subprocess
import sys
import threading

NEW, OLD = sys.argv[1], sys.argv[2]
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MOVETIME = int(sys.argv[4]) if len(sys.argv) > 4 else 100
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
MAX_PLIES = 400

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    "d2d4 d7d5 c2c4",
    "e2e4 c7c5 g1f3 d7d6 d2d4",
    "d2d4 g8f6 c2c4 e7e6 b1c3",
    "e2e4 e7e6 d2d4 d7d5",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 g1f3 g8f6 c1f4",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",
]


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.send("uci")
        self.wait("uciok")
        self.send("isready")
        self.wait("readyok")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith("info string"):
                raise RuntimeError(line.strip())
            if line.startswith(prefix):
                return line.strip()

    @staticmethod
    def position(moves):
        return "position startpos" + (" moves " + " ".join(moves) if moves else "")

    def status(self, moves):
        """Game status. Only engines that implement the non-standard `status` command."""
        self.send(self.position(moves))
        self.send("status")
        return self.wait("status").split()[1]

    def best_move(self, moves):
        self.send(self.position(moves))
        self.send(f"go movetime {MOVETIME}")
        return self.wait("bestmove").split()[1]

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(i):
    new_white = i % 2 == 0
    new, old = Engine(NEW), Engine(OLD)
    white, black = (new, old) if new_white else (old, new)
    moves = OPENINGS[(i // 2) % len(OPENINGS)].split()
    result, reason = 0.5, "move limit"
    try:
        for _ in range(MAX_PLIES):
            white_to_move = len(moves) % 2 == 0
            eng = white if white_to_move else black
            # The NEW build always adjudicates (v0.1 has no `status` command).
            status = new.status(moves)
            mv = eng.best_move(moves) if status == "ongoing" else None
            if status == "checkmate":
                # The side to move has been mated, so the other side won.
                white_won = not white_to_move
                result = 1.0 if (white_won == new_white) else 0.0
                reason = "checkmate"
                break
            if status != "ongoing":
                result, reason = 0.5, status
                break
            if mv == "0000":
                result, reason = 0.5, "no move"
                break
            moves.append(mv)
    finally:
        new.close()
        old.close()
    return i, new_white, result, reason, len(moves)


def elo(p):
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return 400 * math.log10(p / (1 - p))


lock = threading.Lock()
results = []


def run(i):
    r = play_game(i)
    with lock:
        results.append(r)
        i, nw, res, reason, plies = r
        tag = {1.0: "win ", 0.5: "draw", 0.0: "loss"}[res]
        print(
            f"game {i + 1:>3} NEW as {'white' if nw else 'black'}: {tag} ({reason}, {plies} plies)",
            flush=True,
        )
    return r


with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(run, range(GAMES)))

scores = [r[2] for r in results]
n = len(scores)
wins = sum(1 for s in scores if s == 1.0)
draws = sum(1 for s in scores if s == 0.5)
losses = sum(1 for s in scores if s == 0.0)
p = sum(scores) / n
var = sum((s - p) ** 2 for s in scores) / n
se = math.sqrt(var / n)
lo, hi = elo(max(p - 1.96 * se, 0.0)), elo(min(p + 1.96 * se, 1.0))
print()
print(f"NEW vs OLD at {MOVETIME} ms/move: +{wins} ={draws} -{losses}  ({n} games)")
print(f"score {p * 100:.1f}%   Elo difference {elo(p):+.0f}   (95% CI {lo:+.0f} .. {hi:+.0f})")
