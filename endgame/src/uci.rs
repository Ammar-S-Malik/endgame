//! The UCI protocol loop, plus `perft` and `bench` commands.
//!
//! Stdin is read on a separate thread so that `stop` and `quit` can interrupt
//! a running search. Command-line arguments are run as a single command, e.g.
//! `endgame bench` or `endgame perft 5`.

use crate::board::Board;
use crate::movegen::{divide, legal_moves};
use crate::search::{Limits, Searcher};
use crate::types::*;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

const BENCH_FENS: [&str; 6] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "2r3k1/5pp1/p3p2p/1p1pP3/3P4/P1R2PP1/1P4KP/8 w - - 0 1",
];

fn find_move(b: &mut Board, text: &str) -> Option<Move> {
    legal_moves(b).into_iter().find(|m| m.to_uci() == text)
}

fn parse_position(tokens: &[&str]) -> Option<Board> {
    // Both "startpos" and "fen" consume the first token.
    let mut i = 1;
    let mut board = match tokens.first() {
        Some(&"startpos") => Board::startpos(),
        Some(&"fen") => {
            let mut parts = Vec::new();
            while i < tokens.len() && tokens[i] != "moves" {
                parts.push(tokens[i]);
                i += 1;
            }
            Board::from_fen(&parts.join(" ")).ok()?
        }
        _ => return None,
    };
    if tokens.get(i) == Some(&"moves") {
        for text in &tokens[i + 1..] {
            let m = find_move(&mut board, text)?;
            board.make_move(m);
        }
    }
    Some(board)
}

fn handle_go(tokens: &[&str], board: &mut Board, searcher: &mut Searcher) {
    let mut depth = 64;
    let (mut wtime, mut btime, mut winc, mut binc) = (None, None, None, None);
    let (mut movetime, mut movestogo): (Option<u64>, Option<u64>) = (None, None);

    let mut i = 0;
    while i < tokens.len() {
        let value = tokens.get(i + 1).and_then(|s| s.parse::<i64>().ok());
        match tokens[i] {
            "depth" => depth = value.unwrap_or(64) as i32,
            "wtime" => wtime = value.map(|v| v.max(0) as u64),
            "btime" => btime = value.map(|v| v.max(0) as u64),
            "winc" => winc = value.map(|v| v.max(0) as u64),
            "binc" => binc = value.map(|v| v.max(0) as u64),
            "movetime" => movetime = value.map(|v| v.max(1) as u64),
            "movestogo" => movestogo = value.map(|v| v.max(1) as u64),
            _ => {}
        }
        i += 1;
    }

    let (time, inc) = if board.side == WHITE { (wtime, winc) } else { (btime, binc) };
    let (soft, hard) = if let Some(mt) = movetime {
        (u64::MAX, mt.saturating_sub(15).max(1))
    } else if let Some(t) = time {
        let moves_left = movestogo.unwrap_or(30).max(1);
        let inc = inc.unwrap_or(0);
        let soft = (t / moves_left + inc * 3 / 4).min(t * 8 / 10).max(1);
        let hard = (soft * 3).min(t.saturating_sub(50)).max(5);
        (soft, hard)
    } else {
        (u64::MAX, u64::MAX)
    };

    let limits = Limits { depth, soft_ms: soft, hard_ms: hard };
    let result = searcher.search(board, &limits, true);
    println!("bestmove {}", result.best.to_uci());
}

fn run_perft(board: &mut Board, depth: u32) {
    let start = Instant::now();
    let mut total = 0;
    for (m, n) in divide(board, depth) {
        println!("{}: {}", m.to_uci(), n);
        total += n;
    }
    let ms = start.elapsed().as_millis().max(1) as u64;
    println!("\nNodes searched: {total}");
    println!("Time: {ms} ms, {} nps", total * 1000 / ms);
}

fn run_bench(depth: i32, stop: &Arc<AtomicBool>) {
    let start = Instant::now();
    let mut total = 0;
    for fen in BENCH_FENS {
        let mut board = Board::from_fen(fen).expect("bench FEN is valid");
        let mut searcher = Searcher::new(stop.clone());
        let r = searcher.search(&mut board, &Limits::fixed_depth(depth), false);
        println!("{:>10} nodes  best {}  score {:>6}   {fen}", r.nodes, r.best.to_uci(), r.score);
        total += r.nodes;
    }
    let ms = start.elapsed().as_millis().max(1) as u64;
    println!("\n{total} nodes {ms} ms {} nps", total * 1000 / ms);
}

/// Random but roughly balanced opening for testing: `plies` random legal moves from the
/// start position, rejected unless a shallow search says neither side is clearly better.
fn gen_opening(plies: usize, seed: u64, endgame: bool, stop: &Arc<AtomicBool>) -> String {
    let mut searcher = Searcher::new(stop.clone());
    searcher.set_hash_mb(1);
    for attempt in 0..200u64 {
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(attempt.wrapping_mul(0xD1B5_4A32_D192_ED03))
            | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        let mut ok = true;
        for _ in 0..plies {
            let legal = legal_moves(&mut board);
            if legal.is_empty() {
                ok = false;
                break;
            }
            let m = legal[(next() % legal.len() as u64) as usize];
            moves.push(m.to_uci());
            board.make_move(m);
        }
        if !ok || legal_moves(&mut board).is_empty() {
            continue;
        }
        // Endgame starts: keep playing random moves, preferring captures, until little
        // material is left.
        if endgame {
            for _ in 0..80 {
                if (board.occ[0] | board.occ[1]).count_ones() <= 12 {
                    break;
                }
                let legal = legal_moves(&mut board);
                if legal.is_empty() {
                    ok = false;
                    break;
                }
                let caps: Vec<Move> = legal.iter().copied().filter(|m| m.is_capture()).collect();
                let pool = if !caps.is_empty() && next() % 100 < 70 { &caps } else { &legal };
                let m = pool[(next() % pool.len() as u64) as usize];
                moves.push(m.to_uci());
                board.make_move(m);
            }
            if !ok || (board.occ[0] | board.occ[1]).count_ones() > 14 || legal_moves(&mut board).is_empty() {
                continue;
            }
        }
        let r = searcher.search(&mut board, &Limits::fixed_depth(4), false);
        if r.score.abs() <= if endgame { 150 } else { 80 } {
            return moves.join(" ");
        }
    }
    String::new()
}

pub fn run() {
    // The engine ships with a network; `setoption name UseNNUE value false` switches it off.
    if let Err(e) = crate::nnue::load_default() {
        eprintln!("could not load the embedded network: {e}");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<String>();

    if !args.is_empty() {
        let _ = tx.send(args.join(" "));
        let _ = tx.send("quit".to_string());
    } else {
        let stop_flag = stop.clone();
        thread::spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                let text = line.trim().to_string();
                if text == "stop" || text == "quit" {
                    stop_flag.store(true, Ordering::SeqCst);
                }
                if tx.send(text).is_err() {
                    return;
                }
            }
            stop_flag.store(true, Ordering::SeqCst);
            let _ = tx.send("quit".to_string());
        });
    }

    let mut board = Board::startpos();
    let mut searcher = Searcher::new(stop.clone());

    while let Ok(line) = rx.recv() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { continue };
        let rest = &tokens[1..];
        match cmd {
            "uci" => {
                println!("id name Endgame {}", env!("CARGO_PKG_VERSION"));
                println!("id author Ammar Malik");
                println!("option name Hash type spin default 16 min 1 max 1024");
                println!("option name EvalFile type string default <empty>");
                println!("option name UseNNUE type check default true");
                println!("option name Improving type check default true");
                println!("option name ContHist type check default true");
                println!("option name Singular type check default true");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "setoption" => {
                // setoption name Hash value 64
                let lower: Vec<String> = rest.iter().map(|t| t.to_lowercase()).collect();
                if lower.first().map(String::as_str) == Some("name")
                    && lower.get(1).map(String::as_str) == Some("usennue")
                {
                    if lower.get(3).map(String::as_str) == Some("false") {
                        crate::nnue::disable();
                    } else if let Err(e) = crate::nnue::load_default() {
                        println!("info string could not load network: {e}");
                    }
                    board.refresh_nnue();
                } else if lower.first().map(String::as_str) == Some("name")
                    && lower.get(1).map(String::as_str) == Some("evalfile")
                {
                    // Keep the original case of the path.
                    let path = rest.iter().skip_while(|t| t.to_lowercase() != "value").skip(1);
                    let path = path.cloned().collect::<Vec<_>>().join(" ");
                    if !path.is_empty() && path != "<empty>" {
                        match crate::nnue::load_file(&path) {
                            Ok(()) => {
                                board.refresh_nnue();
                                println!("info string loaded network {path}");
                            }
                            Err(e) => println!("info string could not load network: {e}"),
                        }
                    }
                } else if lower.first().map(String::as_str) == Some("name")
                    && lower.get(1).map(String::as_str) == Some("hash")
                {
                    if let Some(mb) = lower.get(3).and_then(|v| v.parse::<usize>().ok()) {
                        searcher.set_hash_mb(mb.clamp(1, 1024));
                    }
                } else if lower.first().map(String::as_str) == Some("name") {
                    // Search feature switches (Improving, ContHist, Singular): "name X value true".
                    if let (Some(n), Some(v)) = (lower.get(1), lower.get(3)) {
                        searcher.set_flag(n.as_str(), v.as_str() == "true" || v.as_str() == "1");
                    }
                }
            }
            "ucinewgame" => {
                board = Board::startpos();
                searcher.new_game();
            }
            "position" => match parse_position(rest) {
                Some(b) => board = b,
                None => println!("info string could not parse position"),
            },
            "go" => {
                handle_go(rest, &mut board, &mut searcher);
                stop.store(false, Ordering::SeqCst);
            }
            // A `stop` that arrives while idle is consumed here.
            "stop" => stop.store(false, Ordering::SeqCst),
            "d" => print!("{}", board.display()),
            // Non-standard helper for the match script: genopening <plies> <seed>
            "genopening" => {
                let plies = rest.first().and_then(|s| s.parse().ok()).unwrap_or(8);
                let seed = rest.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                let endgame = rest.get(2) == Some(&"endgame");
                println!("opening {}", gen_opening(plies, seed, endgame, &stop));
            }
            // Non-standard helper used by the match script to adjudicate games.
            "status" => {
                let status = if legal_moves(&mut board).is_empty() {
                    if board.in_check() { "checkmate" } else { "stalemate" }
                } else if board.halfmove >= 100 {
                    "draw50"
                } else if board.repetitions() >= 2 {
                    "repetition"
                } else if board.insufficient_material() {
                    "insufficient"
                } else {
                    "ongoing"
                };
                println!("status {status}");
            }
            "perft" => {
                let depth = rest.first().and_then(|s| s.parse().ok()).unwrap_or(5);
                run_perft(&mut board, depth);
            }
            "bench" => {
                let depth = rest.first().and_then(|s| s.parse().ok()).unwrap_or(6);
                run_bench(depth, &stop);
            }
            "quit" => break,
            _ => {}
        }
    }
}
