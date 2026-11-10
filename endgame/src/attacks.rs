//! Precomputed attack tables for leapers (knight, king, pawn) and ray-based
//! sliding attacks for bishops, rooks and queens.
//!
//! The sliders use a simple ray walk. It is correct and easy to verify; a
//! magic-bitboard version is a planned speed optimisation.

use crate::types::*;
use std::sync::OnceLock;

pub struct Tables {
    pub knight: [Bb; 64],
    pub king: [Bb; 64],
    /// `pawn[color][sq]` = squares attacked by a pawn of `color` standing on `sq`.
    pub pawn: [[Bb; 64]; 2],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline]
pub fn tables() -> &'static Tables {
    TABLES.get_or_init(build_tables)
}

fn step_attacks(sq: usize, deltas: &[(i32, i32)]) -> Bb {
    let f0 = (sq % 8) as i32;
    let r0 = (sq / 8) as i32;
    let mut bb = 0;
    for &(df, dr) in deltas {
        let f = f0 + df;
        let r = r0 + dr;
        if (0..8).contains(&f) && (0..8).contains(&r) {
            bb |= 1u64 << (r * 8 + f);
        }
    }
    bb
}

fn build_tables() -> Tables {
    const KNIGHT_D: [(i32, i32); 8] =
        [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)];
    const KING_D: [(i32, i32); 8] =
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    let mut t = Tables { knight: [0; 64], king: [0; 64], pawn: [[0; 64]; 2] };
    for sq in 0..64 {
        t.knight[sq] = step_attacks(sq, &KNIGHT_D);
        t.king[sq] = step_attacks(sq, &KING_D);
        t.pawn[WHITE][sq] = step_attacks(sq, &[(1, 1), (-1, 1)]);
        t.pawn[BLACK][sq] = step_attacks(sq, &[(1, -1), (-1, -1)]);
    }
    t
}

fn slide(sq: usize, occ: Bb, dirs: &[(i32, i32); 4]) -> Bb {
    let f0 = (sq % 8) as i32;
    let r0 = (sq / 8) as i32;
    let mut result = 0;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as usize;
            result |= 1u64 << s;
            if (occ >> s) & 1 != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    result
}

#[inline]
pub fn rook_attacks(sq: usize, occ: Bb) -> Bb {
    slide(sq, occ, &[(1, 0), (-1, 0), (0, 1), (0, -1)])
}

#[inline]
pub fn bishop_attacks(sq: usize, occ: Bb) -> Bb {
    slide(sq, occ, &[(1, 1), (1, -1), (-1, 1), (-1, -1)])
}

#[inline]
pub fn queen_attacks(sq: usize, occ: Bb) -> Bb {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}
