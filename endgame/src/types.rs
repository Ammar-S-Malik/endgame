//! Basic types shared across the engine: colors, pieces, squares and moves.
//!
//! Squares are numbered 0..64 with a1 = 0, b1 = 1, ... h1 = 7, a2 = 8, ... h8 = 63.

pub type Bb = u64;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

/// A mailbox entry is `color * 6 + piece_type`, or `EMPTY` for an empty square.
pub const EMPTY: u8 = 12;
/// "No en passant square".
pub const NO_SQ: u8 = 64;

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

// Move flags (4 bits), the standard 16-bit move encoding:
//   0 quiet, 1 double pawn push, 2 king castle, 3 queen castle,
//   4 capture, 5 en passant, 8..=11 promotion (N,B,R,Q), 12..=15 promotion capture.
pub const FLAG_QUIET: u16 = 0;
pub const FLAG_DOUBLE: u16 = 1;
pub const FLAG_KCASTLE: u16 = 2;
pub const FLAG_QCASTLE: u16 = 3;
pub const FLAG_CAPTURE: u16 = 4;
pub const FLAG_EP: u16 = 5;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Move(pub u16);

impl Move {
    /// Sentinel for "no move". (a1 -> a1 is never a real move.)
    pub const NONE: Move = Move(0);

    #[inline]
    pub fn new(from: usize, to: usize, flags: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flags << 12))
    }
    #[inline]
    pub fn from(self) -> usize {
        (self.0 & 63) as usize
    }
    #[inline]
    pub fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }
    #[inline]
    pub fn flags(self) -> u16 {
        self.0 >> 12
    }
    #[inline]
    pub fn is_capture(self) -> bool {
        self.flags() & 4 != 0
    }
    #[inline]
    pub fn is_promo(self) -> bool {
        self.flags() & 8 != 0
    }
    /// Piece type promoted to (KNIGHT..=QUEEN). Only valid if `is_promo()`.
    #[inline]
    pub fn promo_piece(self) -> usize {
        ((self.flags() & 3) + 1) as usize
    }
    #[inline]
    pub fn is_quiet(self) -> bool {
        self.flags() & 12 == 0
    }

    pub fn to_uci(self) -> String {
        if self == Move::NONE {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", sq_name(self.from()), sq_name(self.to()));
        if self.is_promo() {
            s.push(match self.promo_piece() {
                KNIGHT => 'n',
                BISHOP => 'b',
                ROOK => 'r',
                _ => 'q',
            });
        }
        s
    }
}

pub fn sq_name(sq: usize) -> String {
    format!("{}{}", (b'a' + (sq % 8) as u8) as char, sq / 8 + 1)
}

/// A fixed-capacity list of moves (no heap allocation in the search).
pub struct MoveList {
    pub moves: [Move; 256],
    pub scores: [i32; 256],
    pub len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList { moves: [Move::NONE; 256], scores: [0; 256], len: 0 }
    }
    #[inline]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}
