//! Pseudo-legal move generation (legality is checked on make) and perft.

use crate::attacks::*;
use crate::board::Board;
use crate::types::*;

/// Generate pseudo-legal moves for the side to move.
/// With `tactical_only`, only captures and promotions are produced (for quiescence search).
pub fn generate(b: &Board, list: &mut MoveList, tactical_only: bool) {
    let us = b.side;
    let them = us ^ 1;
    let occ_all = b.occ[0] | b.occ[1];
    let own = b.occ[us];
    let enemy = b.occ[them];
    let t = tables();

    let push_dir: i32 = if us == WHITE { 8 } else { -8 };
    let start_rank = if us == WHITE { 1 } else { 6 };
    let promo_rank = if us == WHITE { 6 } else { 1 };

    // --- Pawns -----------------------------------------------------------
    let mut pawns = b.bb[us][PAWN];
    while pawns != 0 {
        let from = pawns.trailing_zeros() as usize;
        pawns &= pawns - 1;
        let rank = from / 8;
        let one = (from as i32 + push_dir) as usize;

        if (occ_all >> one) & 1 == 0 {
            if rank == promo_rank {
                // Queen first so it is tried early.
                list.push(Move::new(from, one, 11));
                list.push(Move::new(from, one, 10));
                list.push(Move::new(from, one, 9));
                list.push(Move::new(from, one, 8));
            } else if !tactical_only {
                list.push(Move::new(from, one, FLAG_QUIET));
                if rank == start_rank {
                    let two = (from as i32 + 2 * push_dir) as usize;
                    if (occ_all >> two) & 1 == 0 {
                        list.push(Move::new(from, two, FLAG_DOUBLE));
                    }
                }
            }
        }

        let mut caps = t.pawn[us][from] & enemy;
        while caps != 0 {
            let to = caps.trailing_zeros() as usize;
            caps &= caps - 1;
            if rank == promo_rank {
                list.push(Move::new(from, to, 15));
                list.push(Move::new(from, to, 14));
                list.push(Move::new(from, to, 13));
                list.push(Move::new(from, to, 12));
            } else {
                list.push(Move::new(from, to, FLAG_CAPTURE));
            }
        }

        if b.ep != NO_SQ && (t.pawn[us][from] >> b.ep) & 1 != 0 {
            list.push(Move::new(from, b.ep as usize, FLAG_EP));
        }
    }

    // --- Knights, bishops, rooks, queens, king ---------------------------
    for pt in KNIGHT..=KING {
        let mut pieces = b.bb[us][pt];
        while pieces != 0 {
            let from = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let attacks = match pt {
                KNIGHT => t.knight[from],
                BISHOP => bishop_attacks(from, occ_all),
                ROOK => rook_attacks(from, occ_all),
                QUEEN => queen_attacks(from, occ_all),
                _ => t.king[from],
            };
            let mut targets = if tactical_only { attacks & enemy } else { attacks & !own };
            while targets != 0 {
                let to = targets.trailing_zeros() as usize;
                targets &= targets - 1;
                let flags = if (enemy >> to) & 1 != 0 { FLAG_CAPTURE } else { FLAG_QUIET };
                list.push(Move::new(from, to, flags));
            }
        }
    }

    // --- Castling --------------------------------------------------------
    if !tactical_only {
        let base = if us == WHITE { 0 } else { 56 };
        let (wk, wq) = if us == WHITE { (CASTLE_WK, CASTLE_WQ) } else { (CASTLE_BK, CASTLE_BQ) };
        let e = base + 4;
        if b.castling & (wk | wq) != 0 && !b.is_attacked(e, them) {
            if b.castling & wk != 0
                && (occ_all >> (base + 5)) & 1 == 0
                && (occ_all >> (base + 6)) & 1 == 0
                && !b.is_attacked(base + 5, them)
                && !b.is_attacked(base + 6, them)
            {
                list.push(Move::new(e, base + 6, FLAG_KCASTLE));
            }
            if b.castling & wq != 0
                && (occ_all >> (base + 1)) & 1 == 0
                && (occ_all >> (base + 2)) & 1 == 0
                && (occ_all >> (base + 3)) & 1 == 0
                && !b.is_attacked(base + 3, them)
                && !b.is_attacked(base + 2, them)
            {
                list.push(Move::new(e, base + 2, FLAG_QCASTLE));
            }
        }
    }
}

/// All fully legal moves in the position.
pub fn legal_moves(b: &mut Board) -> Vec<Move> {
    let mut list = MoveList::new();
    generate(b, &mut list, false);
    let mut out = Vec::with_capacity(list.len);
    for i in 0..list.len {
        let m = list.moves[i];
        if b.make_move_legal(m) {
            b.undo_move();
            out.push(m);
        }
    }
    out
}

/// Count leaf nodes of the legal move tree to `depth`. The standard
/// correctness check for a move generator.
pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate(b, &mut list, false);
    let mut nodes = 0;
    for i in 0..list.len {
        let m = list.moves[i];
        if !b.make_move_legal(m) {
            continue;
        }
        nodes += if depth == 1 { 1 } else { perft(b, depth - 1) };
        b.undo_move();
    }
    nodes
}

/// Perft broken down by root move (handy for hunting bugs against another engine).
pub fn divide(b: &mut Board, depth: u32) -> Vec<(Move, u64)> {
    let mut out = Vec::new();
    for m in legal_moves(b) {
        b.make_move(m);
        let n = if depth <= 1 { 1 } else { perft(b, depth - 1) };
        b.undo_move();
        out.push((m, n));
    }
    out
}
