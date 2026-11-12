//! NNUE consistency tests. These live in their own test binary because loading a network is
//! process-global and would otherwise change the evaluation used by the other tests.

use endgame::board::Board;
use endgame::movegen::legal_moves;
use endgame::nnue::{self, Net, H, INPUTS, KING_BUCKETS, OUT_BUCKETS};

fn random_net(seed: u64) -> Net {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut small = move || ((next() >> 33) % 201) as i16 - 100; // -100..=100
    let mut ft_b = [0i16; H];
    for x in ft_b.iter_mut() {
        *x = small();
    }
    let ft_w: Vec<i16> = (0..INPUTS * H).map(|_| small()).collect();
    let out_w: Vec<i16> = (0..OUT_BUCKETS * 2 * H).map(|_| small()).collect();
    let mut out_b = [0i32; OUT_BUCKETS];
    for (k, x) in out_b.iter_mut().enumerate() {
        *x = 1234 + k as i32;
    }
    Net { ft_w, ft_b, out_w, out_b }
}

/// A network whose weights are a fixed function of their index, so the same network can be
/// built in Python (see the known-answer test below).
fn hash16(i: u64) -> i16 {
    let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ((x >> 40) % 201) as i16 - 100
}

fn synthetic_net() -> Net {
    let ft_w: Vec<i16> = (0..(INPUTS * H) as u64).map(hash16).collect();
    let mut ft_b = [0i16; H];
    for (i, x) in ft_b.iter_mut().enumerate() {
        *x = hash16(10_000_000 + i as u64);
    }
    let out_w: Vec<i16> = (0..(OUT_BUCKETS * 2 * H) as u64).map(|i| hash16(20_000_000 + i)).collect();
    let mut out_b = [0i32; OUT_BUCKETS];
    for (j, x) in out_b.iter_mut().enumerate() {
        *x = hash16(30_000_000 + j as u64) as i32 * 7;
    }
    Net { ft_w, ft_b, out_w, out_b }
}

fn eval_fen(net: &Net, fen: &str) -> i32 {
    let board = Board::from_fen(fen).unwrap();
    let acc = net.refresh(&board);
    net.evaluate(&acc, board.side, (board.occ[0] | board.occ[1]).count_ones())
}

#[test]
fn embedded_default_network_is_valid() {
    assert!(Net::from_bytes(nnue::DEFAULT_NET).is_ok());
    assert_eq!(nnue::DEFAULT_NET.len(), nnue::file_size());
}

#[test]
fn file_round_trip() {
    let net = random_net(7);
    let bytes = net.to_bytes();
    assert_eq!(bytes.len(), nnue::file_size());
    let back = Net::from_bytes(&bytes).unwrap();
    assert_eq!(net.ft_w, back.ft_w);
    assert_eq!(net.ft_b, back.ft_b);
    assert_eq!(net.out_w, back.out_w);
    assert_eq!(net.out_b, back.out_b);
    assert!(Net::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    // An old-format file (magic "ENNU") must be rejected with a clear error, not misread.
    let mut old = bytes.clone();
    old[0..4].copy_from_slice(b"ENNU");
    assert!(Net::from_bytes(&old).is_err());
}

#[test]
fn bucket_functions_stay_in_range() {
    for p in 0..2 {
        for ksq in 0..64 {
            assert!(nnue::king_bucket(p, ksq) < KING_BUCKETS);
        }
    }
    // White king on e1 (rank 0, file e) and black king on e8 are in the same bucket.
    assert_eq!(nnue::king_bucket(0, 4), nnue::king_bucket(1, 60));
    assert_eq!(nnue::king_bucket(0, 4), 1);
    assert_eq!(nnue::king_bucket(0, 0), 0);
    assert_eq!(nnue::king_bucket(0, 63), 7);
    assert_eq!(nnue::output_bucket(2), 0);
    assert_eq!(nnue::output_bucket(32), OUT_BUCKETS - 1);
    assert_eq!(nnue::output_bucket(6), 1);
    for n in 0..40 {
        assert!(nnue::output_bucket(n) < OUT_BUCKETS);
    }
}

/// Evaluations of fixed positions with the synthetic network, computed independently in Python
/// (trainer/train.py's integer_eval_np, itself cross-checked against a scalar reimplementation).
/// This pins down the feature mapping, the king buckets and the output heads.
#[test]
fn known_answers_match_the_python_reference() {
    let net = synthetic_net();
    let cases: &[(&str, i32)] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -793),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1520),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3695),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", -4271),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", -431),
        ("8/5k2/8/8/3K4/8/8/8 b - - 0 1", -1265),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 304),
        ("6k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", -2202),
        ("8/8/8/4k3/8/8/4K3/8 b - - 0 1", -1576),
        ("2kr3r/ppp2ppp/2n5/8/3P4/2N5/PPP2PPP/2KR3R w - - 0 1", 59),
    ];
    for (fen, expected) in cases {
        assert_eq!(eval_fen(&net, fen), *expected, "evaluation of {fen}");
    }
}

#[test]
fn incremental_accumulators_match_refresh_over_random_games() {
    nnue::load_bytes(&random_net(42).to_bytes()).unwrap();
    let net = nnue::net().expect("network loaded");

    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ];
    let mut rng: u64 = 0x1234_5678_9ABC_DEF1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for fen in fens {
        for _ in 0..40 {
            let mut board = Board::from_fen(fen).unwrap();
            assert_eq!(board.acc, net.refresh(&board));
            let mut played = 0;
            for _ in 0..120 {
                let moves = legal_moves(&mut board);
                if moves.is_empty() {
                    break;
                }
                let m = moves[(next() % moves.len() as u64) as usize];
                board.make_move(m);
                played += 1;
                assert_eq!(board.acc, net.refresh(&board), "accumulator drift after {}", m.to_uci());
            }
            for _ in 0..played {
                board.undo_move();
                assert_eq!(board.acc, net.refresh(&board), "accumulator drift after undo");
            }
        }
    }
}
