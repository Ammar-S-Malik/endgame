//! Correctness tests. Run with `cargo test --release` (perft is slow in debug builds).

use endgame::board::Board;
use endgame::movegen::{legal_moves, perft};
use endgame::search::{Limits, Searcher};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Standard perft positions with known node counts per depth
/// (https://www.chessprogramming.org/Perft_Results).
const PERFT_CASES: &[(&str, &str, &[u64])] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8902, 197_281, 4_865_609],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2039, 97_862, 4_085_603],
    ),
    (
        "position 3 (en passant, pins)",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2812, 43_238, 674_624],
    ),
    (
        "position 4 (promotions, castling)",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9467, 422_333],
    ),
    (
        "position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1486, 62_379, 2_103_487],
    ),
    (
        "position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[46, 2079, 89_890, 3_894_594],
    ),
];

#[test]
fn perft_matches_known_counts() {
    for (name, fen, expected) in PERFT_CASES {
        let mut board = Board::from_fen(fen).unwrap();
        for (i, &want) in expected.iter().enumerate() {
            let depth = i as u32 + 1;
            let got = perft(&mut board, depth);
            assert_eq!(got, want, "perft({depth}) mismatch for {name}: {fen}");
        }
    }
}

#[test]
fn fen_round_trip() {
    for (_, fen, _) in PERFT_CASES {
        let board = Board::from_fen(fen).unwrap();
        // Compare the first four fields; move counters may differ in the source FEN.
        let out = board.to_fen();
        let a: Vec<&str> = fen.split(' ').take(4).collect();
        let b: Vec<&str> = out.split(' ').take(4).collect();
        assert_eq!(a, b);
    }
}

/// Play many random legal moves, checking after every move that the incremental
/// hash equals a from-scratch hash, and that undoing everything restores the start.
#[test]
fn hash_and_undo_are_consistent_over_random_games() {
    let mut rng: u64 = 0x1234_5678_9ABC_DEF1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for (_, fen, _) in PERFT_CASES {
        for _game in 0..20 {
            let mut board = Board::from_fen(fen).unwrap();
            let start_fen = board.to_fen();
            let start_hash = board.hash;
            let mut played = 0;
            for _ in 0..120 {
                let moves = legal_moves(&mut board);
                if moves.is_empty() {
                    break;
                }
                let m = moves[(next() % moves.len() as u64) as usize];
                board.make_move(m);
                played += 1;
                assert_eq!(board.hash, board.compute_hash(), "hash drift after {}", m.to_uci());
                assert_eq!(
                    (board.mg, board.eg, board.phase),
                    endgame::eval::scratch_sums(&board),
                    "incremental eval drift after {}",
                    m.to_uci()
                );
            }
            for _ in 0..played {
                board.undo_move();
            }
            assert_eq!(board.to_fen(), start_fen);
            assert_eq!(board.hash, start_hash);
        }
    }
}

fn best_move(fen: &str, depth: i32) -> String {
    let mut board = Board::from_fen(fen).unwrap();
    let mut searcher = Searcher::new(Arc::new(AtomicBool::new(false)));
    let result = searcher.search(&mut board, &Limits::fixed_depth(depth), false);
    result.best.to_uci()
}

#[test]
fn finds_back_rank_mate_in_one() {
    // Ra8# is the only mate.
    assert_eq!(best_move("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 3), "a1a8");
}

#[test]
fn reports_mate_scores() {
    // Rook ladder: Rb6-b8# is mate in one, so the score must be MATE - 1 ply.
    let mut board = Board::from_fen("7k/R7/1R6/8/8/8/8/K7 w - - 0 1").unwrap();
    let mut searcher = Searcher::new(Arc::new(AtomicBool::new(false)));
    let result = searcher.search(&mut board, &Limits::fixed_depth(5), false);
    assert_eq!(result.best.to_uci(), "b6b8");
    assert_eq!(result.score, 31_000 - 1);
}

#[test]
fn takes_a_free_queen() {
    // Black queen on d5 is hanging to the pawn on e4.
    assert_eq!(best_move("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", 4), "e4d5");
}

#[test]
fn avoids_stalemating_when_winning() {
    // White is up a queen against a cornered king; many queen moves stalemate. The
    // engine must pick a move that either mates or leaves Black a legal reply.
    let mv = best_move("k7/8/1K6/8/8/6Q1/8/8 w - - 0 1", 5);
    let mut board = Board::from_fen("k7/8/1K6/8/8/6Q1/8/8 w - - 0 1").unwrap();
    let m = legal_moves(&mut board).into_iter().find(|m| m.to_uci() == mv).unwrap();
    board.make_move(m);
    // Either checkmate, or black still has a legal move (not stalemate).
    let black_moves = legal_moves(&mut board).len();
    assert!(black_moves > 0 || board.in_check(), "engine stalemated the opponent with {mv}");
}
