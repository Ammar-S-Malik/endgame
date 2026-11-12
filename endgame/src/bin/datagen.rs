//! Self-play training-data generator for the NNUE trainer.
//!
//! usage: datagen --games N --depth D --seed S --threads T --out PREFIX [--net FILE]
//!
//! Each game starts with a few random moves, then both sides play fixed-depth searches.
//! Every quiet, non-check position is recorded as one line:
//!     <fen>;<search score in centipawns, White's point of view>;<result: 1, 0.5 or 0, White's POV>
//! One output file per thread: PREFIX.<thread>.txt

use endgame::board::Board;
use endgame::movegen::legal_moves;
use endgame::search::{Limits, Searcher, MATE};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const OPENING_PLIES: usize = 8;
const MAX_PLIES: usize = 320;
// Only adjudicate hopelessly decided games, so that plenty of endgame positions get played out.
const WIN_CP: i32 = 1500;
const WIN_COUNT: usize = 8;
/// Fraction (out of 100) of games that start from a reduced-material, endgame-like position.
const ENDGAME_START_PERCENT: u64 = 25;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Plays one game; returns (positions as (fen, white-POV score), result for White).
fn play_game(searcher: &mut Searcher, rng: &mut Rng, depth: i32) -> Option<(Vec<(String, i32)>, f64)> {
    let mut board = Board::startpos();
    for _ in 0..OPENING_PLIES {
        let moves = legal_moves(&mut board);
        if moves.is_empty() {
            return None;
        }
        let m = moves[(rng.next() % moves.len() as u64) as usize];
        board.make_move(m);
    }
    // Some games start from an endgame-like position: keep playing random moves, preferring
    // captures, until little material is left (so the data covers sparse positions too).
    if rng.next() % 100 < ENDGAME_START_PERCENT {
        for _ in 0..80 {
            if (board.occ[0] | board.occ[1]).count_ones() <= 12 {
                break;
            }
            let moves = legal_moves(&mut board);
            if moves.is_empty() {
                return None;
            }
            let captures: Vec<_> = moves.iter().copied().filter(|m| m.is_capture()).collect();
            let pool = if !captures.is_empty() && rng.next() % 100 < 70 { &captures } else { &moves };
            let m = pool[(rng.next() % pool.len() as u64) as usize];
            board.make_move(m);
        }
    }
    // Reject wildly unbalanced random openings.
    let probe = searcher.search(&mut board, &Limits::fixed_depth(4), false);
    if probe.score.abs() > 300 || legal_moves(&mut board).is_empty() {
        return None;
    }

    let mut positions: Vec<(String, i32)> = Vec::new();
    let mut streak = 0usize;
    let mut streak_sign = 0i32;
    let limits = Limits::fixed_depth(depth);
    let result: f64;
    let mut ply = 0;
    loop {
        let moves = legal_moves(&mut board);
        if moves.is_empty() {
            result = if board.in_check() {
                if board.side == 0 { 0.0 } else { 1.0 } // side to move is mated
            } else {
                0.5
            };
            break;
        }
        if board.halfmove >= 100 || board.repetitions() >= 2 || board.insufficient_material() || ply >= MAX_PLIES {
            result = 0.5;
            break;
        }
        let r = searcher.search(&mut board, &limits, false);
        let white_score = if board.side == 0 { r.score } else { -r.score };

        // Record only quiet positions (not in check, best move not a capture or promotion).
        let best = r.best;
        if !board.in_check() && best.is_quiet() && r.score.abs() < MATE - 200 {
            positions.push((board.to_fen(), white_score));
        }

        // Adjudicate clearly decided games.
        if white_score.abs() >= WIN_CP {
            let sign = white_score.signum();
            streak = if sign == streak_sign { streak + 1 } else { 1 };
            streak_sign = sign;
            if streak >= WIN_COUNT {
                result = if sign > 0 { 1.0 } else { 0.0 };
                break;
            }
        } else {
            streak = 0;
            streak_sign = 0;
        }
        board.make_move(best);
        ply += 1;
    }
    Some((positions, result))
}

fn main() {
    let mut games = 1000usize;
    let mut depth = 5i32;
    let mut seed = 1u64;
    let mut threads = 2usize;
    let mut out = String::from("data");
    let mut net: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--games" => games = v.parse().expect("--games"),
            "--depth" => depth = v.parse().expect("--depth"),
            "--seed" => seed = v.parse().expect("--seed"),
            "--threads" => threads = v.parse().expect("--threads"),
            "--out" => out = v,
            "--net" => net = Some(v),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    if let Some(path) = &net {
        endgame::nnue::load_file(path).expect("could not load network");
        eprintln!("using network {path}");
    }

    let counter = Arc::new(AtomicUsize::new(0));
    let positions_total = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let counter = counter.clone();
            let positions_total = positions_total.clone();
            let path = format!("{out}.{t}.txt");
            thread::spawn(move || {
                let mut searcher = Searcher::new(Arc::new(AtomicBool::new(false)));
                searcher.set_hash_mb(16);
                let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((t as u64 + 1) << 32) | 1);
                let mut w = BufWriter::new(File::create(&path).expect("create output"));
                loop {
                    let g = counter.fetch_add(1, Ordering::SeqCst);
                    if g >= games {
                        break;
                    }
                    searcher.new_game();
                    if let Some((positions, result)) = play_game(&mut searcher, &mut rng, depth) {
                        for (fen, score) in &positions {
                            writeln!(w, "{fen};{score};{result}").unwrap();
                        }
                        positions_total.fetch_add(positions.len(), Ordering::SeqCst);
                    }
                    if (g + 1) % 100 == 0 {
                        eprintln!(
                            "{} games, {} positions, {:.0}s",
                            g + 1,
                            positions_total.load(Ordering::SeqCst),
                            start.elapsed().as_secs_f64()
                        );
                    }
                }
                w.flush().unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    eprintln!(
        "done: {} positions in {:.0}s",
        positions_total.load(Ordering::SeqCst),
        start.elapsed().as_secs_f64()
    );
}
