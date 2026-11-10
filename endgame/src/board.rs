//! The board: bitboards + a mailbox for fast piece lookup, FEN parsing,
//! incremental Zobrist hashing, and make/undo of moves.

use crate::attacks::*;
use crate::eval::{scratch_sums, EG_TABLE, MG_TABLE, PHASE_WEIGHT};
use crate::types::*;
use std::sync::OnceLock;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const PIECE_CHARS: &str = "PNBRQKpnbrqk";

// ---------------------------------------------------------------------------
// Zobrist keys
// ---------------------------------------------------------------------------

pub struct Zobrist {
    pub piece: [[u64; 64]; 12],
    pub castle: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}

static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

pub fn zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        // splitmix64, fixed seed: deterministic hashes across runs.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut z = Zobrist { piece: [[0; 64]; 12], castle: [0; 16], ep: [0; 8], side: 0 };
        for p in 0..12 {
            for sq in 0..64 {
                z.piece[p][sq] = next();
            }
        }
        for c in 0..16 {
            z.castle[c] = next();
        }
        for f in 0..8 {
            z.ep[f] = next();
        }
        z.side = next();
        z
    })
}

/// Which castling rights survive a move from or to `sq`.
#[inline]
fn castle_keep(sq: usize) -> u8 {
    match sq {
        0 => !CASTLE_WQ,
        4 => !(CASTLE_WK | CASTLE_WQ),
        7 => !CASTLE_WK,
        56 => !CASTLE_BQ,
        60 => !(CASTLE_BK | CASTLE_BQ),
        63 => !CASTLE_BK,
        _ => 0xFF,
    }
}

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Undo {
    mv: Move,
    captured: u8,
    castling: u8,
    ep: u8,
    halfmove: u32,
    hash: u64,
}

#[derive(Clone)]
pub struct Board {
    /// `bb[color][piece_type]`
    pub bb: [[Bb; 6]; 2],
    pub occ: [Bb; 2],
    pub mailbox: [u8; 64],
    pub side: usize,
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u32,
    pub fullmove: u32,
    pub hash: u64,
    /// Incrementally maintained evaluation terms (White's point of view), see `eval.rs`.
    pub mg: i32,
    pub eg: i32,
    pub phase: i32,
    undo: Vec<Undo>,
    /// Hash of every position since the start of the game (current one last).
    keys: Vec<u64>,
}

impl Board {
    fn empty() -> Board {
        Board {
            bb: [[0; 6]; 2],
            occ: [0; 2],
            mailbox: [EMPTY; 64],
            side: WHITE,
            castling: 0,
            ep: NO_SQ,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            mg: 0,
            eg: 0,
            phase: 0,
            undo: Vec::with_capacity(512),
            keys: Vec::with_capacity(512),
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen(STARTPOS).expect("start position FEN is valid")
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("bad FEN: {fen}"));
        }
        let mut b = Board::empty();
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += c.to_digit(10).unwrap() as i32,
                _ => {
                    let code = PIECE_CHARS
                        .find(c)
                        .ok_or_else(|| format!("bad piece '{c}' in FEN"))?;
                    if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                        return Err("FEN square out of range".to_string());
                    }
                    let sq = (rank * 8 + file) as usize;
                    let (color, pt) = (code / 6, code % 6);
                    b.bb[color][pt] |= 1u64 << sq;
                    b.occ[color] |= 1u64 << sq;
                    b.mailbox[sq] = code as u8;
                    file += 1;
                }
            }
        }
        b.side = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            other => return Err(format!("bad side to move '{other}'")),
        };
        if let Some(cs) = parts.get(2) {
            for c in cs.chars() {
                match c {
                    'K' => b.castling |= CASTLE_WK,
                    'Q' => b.castling |= CASTLE_WQ,
                    'k' => b.castling |= CASTLE_BK,
                    'q' => b.castling |= CASTLE_BQ,
                    _ => {}
                }
            }
        }
        if let Some(ep) = parts.get(3) {
            let bytes = ep.as_bytes();
            if bytes.len() == 2 && bytes[0] >= b'a' && bytes[0] <= b'h' && bytes[1] >= b'1' && bytes[1] <= b'8' {
                b.ep = (bytes[1] - b'1') * 8 + (bytes[0] - b'a');
            }
        }
        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        b.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        b.hash = b.compute_hash();
        let (mg, eg, phase) = scratch_sums(&b);
        b.mg = mg;
        b.eg = eg;
        b.phase = phase;
        b.keys.push(b.hash);
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let p = self.mailbox[rank * 8 + file];
                if p == EMPTY {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    s.push(PIECE_CHARS.as_bytes()[p as usize] as char);
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.side == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        } else {
            if self.castling & CASTLE_WK != 0 {
                s.push('K');
            }
            if self.castling & CASTLE_WQ != 0 {
                s.push('Q');
            }
            if self.castling & CASTLE_BK != 0 {
                s.push('k');
            }
            if self.castling & CASTLE_BQ != 0 {
                s.push('q');
            }
        }
        s.push(' ');
        if self.ep == NO_SQ {
            s.push('-');
        } else {
            s.push_str(&sq_name(self.ep as usize));
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    pub fn display(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            s.push_str(&format!("{} ", rank + 1));
            for file in 0..8 {
                let p = self.mailbox[rank * 8 + file];
                let c = if p == EMPTY { '.' } else { PIECE_CHARS.as_bytes()[p as usize] as char };
                s.push(c);
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str("  a b c d e f g h\n");
        s.push_str(&format!("FEN: {}\n", self.to_fen()));
        s
    }

    /// Recompute the hash from scratch (used at setup and in tests).
    pub fn compute_hash(&self) -> u64 {
        let z = zobrist();
        let mut h = 0u64;
        for sq in 0..64 {
            let p = self.mailbox[sq];
            if p != EMPTY {
                h ^= z.piece[p as usize][sq];
            }
        }
        if self.side == BLACK {
            h ^= z.side;
        }
        h ^= z.castle[self.castling as usize];
        if self.ep != NO_SQ {
            h ^= z.ep[(self.ep % 8) as usize];
        }
        h
    }

    #[inline]
    pub fn king_sq(&self, color: usize) -> usize {
        self.bb[color][KING].trailing_zeros() as usize
    }

    /// Is `sq` attacked by any piece of color `by`?
    #[inline]
    pub fn is_attacked(&self, sq: usize, by: usize) -> bool {
        let t = tables();
        let occ = self.occ[0] | self.occ[1];
        let theirs = &self.bb[by];
        if t.pawn[by ^ 1][sq] & theirs[PAWN] != 0 {
            return true;
        }
        if t.knight[sq] & theirs[KNIGHT] != 0 {
            return true;
        }
        if t.king[sq] & theirs[KING] != 0 {
            return true;
        }
        if bishop_attacks(sq, occ) & (theirs[BISHOP] | theirs[QUEEN]) != 0 {
            return true;
        }
        rook_attacks(sq, occ) & (theirs[ROOK] | theirs[QUEEN]) != 0
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.is_attacked(self.king_sq(self.side), self.side ^ 1)
    }

    /// Has the current position occurred before (within the reversible window)?
    pub fn is_repetition(&self) -> bool {
        let n = self.keys.len();
        let max_back = (self.halfmove as usize).min(n - 1);
        let mut i = 2;
        while i <= max_back {
            if self.keys[n - 1 - i] == self.hash {
                return true;
            }
            i += 2;
        }
        false
    }

    /// How many times has the current position already occurred in the game?
    pub fn repetitions(&self) -> usize {
        let n = self.keys.len();
        let max_back = (self.halfmove as usize).min(n - 1);
        let mut count = 0;
        let mut i = 2;
        while i <= max_back {
            if self.keys[n - 1 - i] == self.hash {
                count += 1;
            }
            i += 2;
        }
        count
    }

    /// K vs K, or K + a single minor piece vs K: neither side can ever mate.
    pub fn insufficient_material(&self) -> bool {
        let heavy = self.bb[0][PAWN]
            | self.bb[1][PAWN]
            | self.bb[0][ROOK]
            | self.bb[1][ROOK]
            | self.bb[0][QUEEN]
            | self.bb[1][QUEEN];
        if heavy != 0 {
            return false;
        }
        let minors = (self.bb[0][KNIGHT] | self.bb[0][BISHOP] | self.bb[1][KNIGHT] | self.bb[1][BISHOP])
            .count_ones();
        minors <= 1
    }

    /// Any non-pawn, non-king material for the side to move? (Used to avoid
    /// null-move pruning in zugzwang-prone endgames.)
    #[inline]
    pub fn has_non_pawn_material(&self, color: usize) -> bool {
        self.bb[color][KNIGHT] | self.bb[color][BISHOP] | self.bb[color][ROOK] | self.bb[color][QUEEN] != 0
    }

    #[inline]
    fn add_piece(&mut self, color: usize, pt: usize, sq: usize) {
        let bit = 1u64 << sq;
        self.bb[color][pt] |= bit;
        self.occ[color] |= bit;
        self.mailbox[sq] = (color * 6 + pt) as u8;
        self.hash ^= zobrist().piece[color * 6 + pt][sq];
        self.mg += MG_TABLE[color * 6 + pt][sq];
        self.eg += EG_TABLE[color * 6 + pt][sq];
        self.phase += PHASE_WEIGHT[pt];
    }

    #[inline]
    fn remove_piece(&mut self, color: usize, pt: usize, sq: usize) {
        let bit = 1u64 << sq;
        self.bb[color][pt] &= !bit;
        self.occ[color] &= !bit;
        self.mailbox[sq] = EMPTY;
        self.hash ^= zobrist().piece[color * 6 + pt][sq];
        self.mg -= MG_TABLE[color * 6 + pt][sq];
        self.eg -= EG_TABLE[color * 6 + pt][sq];
        self.phase -= PHASE_WEIGHT[pt];
    }

    /// Play a (pseudo-legal) move. Does not check that it leaves the king safe.
    pub fn make_move(&mut self, m: Move) {
        let z = zobrist();
        let us = self.side;
        let them = us ^ 1;
        let from = m.from();
        let to = m.to();
        let flags = m.flags();
        let pt = (self.mailbox[from] % 6) as usize;

        let ep_cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to + 8 };
        let captured = if flags == FLAG_EP {
            self.mailbox[ep_cap_sq]
        } else if m.is_capture() {
            self.mailbox[to]
        } else {
            EMPTY
        };

        self.undo.push(Undo {
            mv: m,
            captured,
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            hash: self.hash,
        });

        // Clear the old en passant square from the hash.
        if self.ep != NO_SQ {
            self.hash ^= z.ep[(self.ep % 8) as usize];
        }
        self.ep = NO_SQ;

        // Remove the captured piece.
        if flags == FLAG_EP {
            self.remove_piece(them, PAWN, ep_cap_sq);
        } else if captured != EMPTY {
            self.remove_piece(them, (captured % 6) as usize, to);
        }

        // Move the piece (promoting if needed).
        self.remove_piece(us, pt, from);
        let placed = if m.is_promo() { m.promo_piece() } else { pt };
        self.add_piece(us, placed, to);

        // Move the rook when castling.
        if flags == FLAG_KCASTLE {
            self.remove_piece(us, ROOK, to + 1);
            self.add_piece(us, ROOK, to - 1);
        } else if flags == FLAG_QCASTLE {
            self.remove_piece(us, ROOK, to - 2);
            self.add_piece(us, ROOK, to + 1);
        }

        self.halfmove = if pt == PAWN || captured != EMPTY { 0 } else { self.halfmove + 1 };

        if flags == FLAG_DOUBLE {
            self.ep = ((from + to) / 2) as u8;
            self.hash ^= z.ep[(self.ep % 8) as usize];
        }

        // Update castling rights.
        self.hash ^= z.castle[self.castling as usize];
        self.castling &= castle_keep(from) & castle_keep(to);
        self.hash ^= z.castle[self.castling as usize];

        self.side = them;
        self.hash ^= z.side;
        if us == BLACK {
            self.fullmove += 1;
        }
        self.keys.push(self.hash);
    }

    /// Undo the last move played with `make_move`.
    pub fn undo_move(&mut self) {
        let u = self.undo.pop().expect("undo_move with empty history");
        self.keys.pop();
        let them = self.side;
        let us = them ^ 1;
        self.side = us;
        let m = u.mv;
        let from = m.from();
        let to = m.to();
        let flags = m.flags();

        let placed_pt = (self.mailbox[to] % 6) as usize;
        self.remove_piece(us, placed_pt, to);
        let original = if m.is_promo() { PAWN } else { placed_pt };
        self.add_piece(us, original, from);

        if flags == FLAG_EP {
            let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
            self.add_piece(them, PAWN, cap_sq);
        } else if u.captured != EMPTY {
            self.add_piece(them, (u.captured % 6) as usize, to);
        }

        if flags == FLAG_KCASTLE {
            self.remove_piece(us, ROOK, to - 1);
            self.add_piece(us, ROOK, to + 1);
        } else if flags == FLAG_QCASTLE {
            self.remove_piece(us, ROOK, to + 1);
            self.add_piece(us, ROOK, to - 2);
        }

        self.castling = u.castling;
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.hash = u.hash;
        if us == BLACK {
            self.fullmove -= 1;
        }
    }

    /// Make the move only if it is legal. Returns whether it was played.
    #[inline]
    pub fn make_move_legal(&mut self, m: Move) -> bool {
        self.make_move(m);
        let mover = self.side ^ 1;
        if self.is_attacked(self.king_sq(mover), self.side) {
            self.undo_move();
            false
        } else {
            true
        }
    }

    /// Pass the turn (for null-move pruning).
    pub fn make_null(&mut self) {
        let z = zobrist();
        self.undo.push(Undo {
            mv: Move::NONE,
            captured: EMPTY,
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            hash: self.hash,
        });
        if self.ep != NO_SQ {
            self.hash ^= z.ep[(self.ep % 8) as usize];
        }
        self.ep = NO_SQ;
        self.halfmove += 1;
        self.side ^= 1;
        self.hash ^= z.side;
        self.keys.push(self.hash);
    }

    pub fn undo_null(&mut self) {
        let u = self.undo.pop().expect("undo_null with empty history");
        self.keys.pop();
        self.side ^= 1;
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.hash = u.hash;
    }
}
