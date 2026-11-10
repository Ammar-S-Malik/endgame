//! Static evaluation: material + piece-square tables, with a tapered king.
//! Scores are in centipawns from the point of view of the side to move.
//!
//! Tables are the well-known "Simplified Evaluation Function" values, written
//! from White's perspective with rank 8 on the top row.

use crate::board::Board;
use crate::types::*;

pub const VALUE: [i32; 6] = [100, 320, 330, 500, 900, 0];
const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];

#[rustfmt::skip]
const PST: [[i32; 64]; 5] = [
    // Pawn
    [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    // Knight
    [
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    // Bishop
    [
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    // Rook
    [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    // Queen
    [
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
         0,  0,  5,  5,  5,  5,  0, -5,
       -10,  5,  5,  5,  5,  5,  0,-10,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20,
    ],
];

#[rustfmt::skip]
const KING_MG: [i32; 64] = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_EG: [i32; 64] = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
];

/// Per-(piece code, square) score contribution in centipawns, from White's point of
/// view: positive for White pieces, negative for Black pieces. `code = color * 6 + type`.
/// The board adds/subtracts these as pieces move, so evaluation is O(1).
const fn build_table(endgame: bool) -> [[i32; 64]; 12] {
    let mut t = [[0i32; 64]; 12];
    let mut code = 0;
    while code < 12 {
        let color = code / 6;
        let pt = code % 6;
        let mut sq = 0;
        while sq < 64 {
            // Tables are written rank-8-first, so White flips the rank.
            let idx = if color == 0 { sq ^ 56 } else { sq };
            let pst = if pt == 5 {
                if endgame {
                    KING_EG[idx]
                } else {
                    KING_MG[idx]
                }
            } else {
                PST[pt][idx]
            };
            let v = VALUE[pt] + pst;
            t[code][sq] = if color == 0 { v } else { -v };
            sq += 1;
        }
        code += 1;
    }
    t
}

pub static MG_TABLE: [[i32; 64]; 12] = build_table(false);
pub static EG_TABLE: [[i32; 64]; 12] = build_table(true);

/// Game-phase weight of a piece type (knights/bishops 1, rooks 2, queens 4; 24 total per full board).
pub const PHASE_WEIGHT: [i32; 6] = PHASE;

/// Recompute (middlegame, endgame, phase) sums from scratch. Used when setting up a
/// position and by tests to check the incremental values never drift.
pub fn scratch_sums(b: &Board) -> (i32, i32, i32) {
    let (mut mg, mut eg, mut phase) = (0, 0, 0);
    for sq in 0..64 {
        let p = b.mailbox[sq];
        if p != EMPTY {
            mg += MG_TABLE[p as usize][sq];
            eg += EG_TABLE[p as usize][sq];
            phase += PHASE[(p % 6) as usize];
        }
    }
    (mg, eg, phase)
}

/// Static evaluation in centipawns from the side to move's point of view.
#[inline]
pub fn evaluate(b: &Board) -> i32 {
    let phase = b.phase.min(24);
    let score = (b.mg * phase + b.eg * (24 - phase)) / 24;
    if b.side == WHITE {
        score
    } else {
        -score
    }
}
