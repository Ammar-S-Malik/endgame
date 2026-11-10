//! Search: iterative deepening, negamax alpha-beta, quiescence search,
//! null-move pruning, check extensions, and move ordering (MVV-LVA,
//! killer moves, history heuristic).

use crate::board::Board;
use crate::eval::{evaluate, VALUE};
use crate::movegen::{generate, legal_moves};
use crate::tt::{TranspositionTable, FLAG_EXACT, FLAG_LOWER, FLAG_UPPER};
use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const INF: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MAX_PLY: usize = 128;
const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

#[derive(Clone, Copy)]
pub struct Limits {
    pub depth: i32,
    /// Don't start another iteration once about half of this has elapsed. `u64::MAX` = no limit.
    pub soft_ms: u64,
    /// Abort the search at this point. `u64::MAX` = no limit.
    pub hard_ms: u64,
}

impl Limits {
    pub fn fixed_depth(depth: i32) -> Limits {
        Limits { depth, soft_ms: u64::MAX, hard_ms: u64::MAX }
    }
}

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

pub struct Searcher {
    stop: Arc<AtomicBool>,
    tt: TranspositionTable,
    nodes: u64,
    start: Instant,
    hard_ms: u64,
    aborted: bool,
    iter_depth: i32,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
}

fn score_str(score: i32) -> String {
    if score.abs() >= MATE_BOUND {
        let plies = MATE - score.abs();
        let moves = (plies + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {score}")
    }
}

/// Mate scores are stored relative to the node, not the root, so they stay valid
/// when the same position is reached at a different ply.
#[inline]
fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s + ply as i32
    } else if s <= -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline]
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s - ply as i32
    } else if s <= -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

impl Searcher {
    pub fn set_hash_mb(&mut self, mb: usize) {
        self.tt.resize(mb);
    }

    /// Forget everything learned about the previous game.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.history = [[[0; 64]; 64]; 2];
    }

    pub fn new(stop: Arc<AtomicBool>) -> Box<Searcher> {
        Box::new(Searcher {
            stop,
            tt: TranspositionTable::new(16),
            nodes: 0,
            start: Instant::now(),
            hard_ms: u64::MAX,
            aborted: false,
            iter_depth: 0,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            pv: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
        })
    }

    pub fn search(&mut self, b: &mut Board, limits: &Limits, report: bool) -> SearchResult {
        self.nodes = 0;
        self.start = Instant::now();
        self.hard_ms = limits.hard_ms;
        self.aborted = false;
        self.tt.new_search();
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        for side in self.history.iter_mut() {
            for from in side.iter_mut() {
                for h in from.iter_mut() {
                    *h /= 2;
                }
            }
        }

        let root_moves = legal_moves(b);
        let mut result = SearchResult {
            best: root_moves.first().copied().unwrap_or(Move::NONE),
            score: 0,
            depth: 0,
            nodes: 0,
        };
        // A forced move in a timed game needs no thinking.
        if root_moves.len() <= 1 && limits.soft_ms != u64::MAX {
            return result;
        }

        let max_depth = limits.depth.min(MAX_PLY as i32 - 2);
        for depth in 1..=max_depth {
            self.iter_depth = depth;
            let score = self.negamax(b, depth, -INF, INF, 0, true);
            if self.aborted {
                break;
            }
            if self.pv_len[0] > 0 {
                result.best = self.pv[0][0];
            }
            result.score = score;
            result.depth = depth;
            result.nodes = self.nodes;

            let elapsed = self.start.elapsed().as_millis() as u64;
            if report {
                let pv: Vec<String> =
                    self.pv[0][..self.pv_len[0]].iter().map(|m| m.to_uci()).collect();
                println!(
                    "info depth {} score {} nodes {} nps {} time {} pv {}",
                    depth,
                    score_str(score),
                    self.nodes,
                    self.nodes * 1000 / elapsed.max(1),
                    elapsed,
                    pv.join(" ")
                );
            }
            if limits.soft_ms != u64::MAX && elapsed * 2 >= limits.soft_ms {
                break;
            }
            // A found mate in N plies is proven once we have searched N plies.
            if score >= MATE_BOUND && depth >= MATE - score {
                break;
            }
        }
        result.nodes = self.nodes;
        result
    }

    #[inline]
    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        // Depth 1 always completes so we always have a move to play.
        if self.iter_depth > 1 && (self.nodes & 2047) == 0 {
            if self.stop.load(Ordering::Relaxed)
                || self.start.elapsed().as_millis() as u64 >= self.hard_ms
            {
                self.aborted = true;
                return true;
            }
        }
        false
    }

    fn score_moves(&self, b: &Board, list: &mut MoveList, ply: usize, tt_move: Move) {
        let us = b.side;
        for i in 0..list.len {
            let m = list.moves[i];
            if m == tt_move {
                list.scores[i] = 1_000_000;
                continue;
            }
            let mut s = 0;
            if m.is_capture() {
                let victim = if m.flags() == FLAG_EP {
                    PAWN
                } else {
                    (b.mailbox[m.to()] % 6) as usize
                };
                let attacker = (b.mailbox[m.from()] % 6) as usize;
                // Most valuable victim, least valuable attacker.
                s += 100_000 + VALUE[victim] * 10 - VALUE[attacker] / 10;
            }
            if m.is_promo() {
                s += 90_000 + m.promo_piece() as i32 * 1000;
            }
            if m.is_quiet() {
                s = if m == self.killers[ply][0] {
                    80_000
                } else if m == self.killers[ply][1] {
                    79_000
                } else {
                    self.history[us][m.from()][m.to()]
                };
            }
            list.scores[i] = s;
        }
    }

    #[inline]
    fn pick_best(list: &mut MoveList, start: usize) {
        let mut best = start;
        let mut best_s = list.scores[start];
        for j in start + 1..list.len {
            if list.scores[j] > best_s {
                best = j;
                best_s = list.scores[j];
            }
        }
        if best != start {
            list.moves.swap(start, best);
            list.scores.swap(start, best);
        }
    }

    fn negamax(
        &mut self,
        b: &mut Board,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        self.pv_len[ply] = 0;
        if self.check_abort() {
            return 0;
        }
        if ply > 0 && (b.halfmove >= 100 || b.is_repetition()) {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(b);
        }

        let in_check = b.in_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch(b, alpha, beta, ply);
        }
        self.nodes += 1;

        // Transposition table: reuse earlier results for this position. We only cut off
        // in non-PV nodes (null window) so the principal variation stays intact.
        let is_pv = beta - alpha > 1;
        let mut tt_move = Move::NONE;
        if let Some(e) = self.tt.probe(b.hash) {
            tt_move = Move(e.mv);
            if ply > 0 && !is_pv && e.depth as i32 >= depth {
                let s = score_from_tt(e.score as i32, ply);
                match e.flag {
                    FLAG_EXACT => return s,
                    FLAG_LOWER if s >= beta => return s,
                    FLAG_UPPER if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        // Null-move pruning: if we pass and still beat beta, this node is very likely a cutoff.
        if allow_null
            && !in_check
            && depth >= 3
            && ply > 0
            && beta < MATE_BOUND
            && b.has_non_pawn_material(b.side)
            && evaluate(b) >= beta
        {
            let r = 2 + (depth >= 6) as i32;
            b.make_null();
            let score = -self.negamax(b, depth - 1 - r, -beta, -beta + 1, ply + 1, false);
            b.undo_null();
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        let mut list = MoveList::new();
        generate(b, &mut list, false);
        self.score_moves(b, &mut list, ply, tt_move);

        let us = b.side;
        let orig_alpha = alpha;
        let mut legal = 0;
        let mut best = -INF;
        let mut best_move = Move::NONE;

        for i in 0..list.len {
            Self::pick_best(&mut list, i);
            let m = list.moves[i];
            if !b.make_move_legal(m) {
                continue;
            }
            legal += 1;

            // Principal variation search: the first move gets the full window; later
            // moves get a null window and are only re-searched if they beat alpha.
            let score = if legal == 1 {
                -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, true)
            } else {
                let mut s = -self.negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                if s > alpha && s < beta && !self.aborted {
                    s = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, true);
                }
                s
            };
            b.undo_move();
            if self.aborted {
                return 0;
            }

            if score > best {
                best = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    // Update the principal variation.
                    self.pv[ply][0] = m;
                    let n = self.pv_len[ply + 1];
                    for j in 0..n {
                        self.pv[ply][j + 1] = self.pv[ply + 1][j];
                    }
                    self.pv_len[ply] = n + 1;

                    if alpha >= beta {
                        if m.is_quiet() {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            let bonus = (depth * depth).min(400);
                            let h = &mut self.history[us][m.from()][m.to()];
                            *h += bonus - *h * bonus / 16384;
                        }
                        break;
                    }
                }
            }
        }

        if legal == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let flag = if best >= beta {
            FLAG_LOWER
        } else if best > orig_alpha {
            FLAG_EXACT
        } else {
            FLAG_UPPER
        };
        // A fail-low node has no meaningful best move.
        let store_move = if flag == FLAG_UPPER { Move::NONE } else { best_move };
        self.tt.store(b.hash, store_move, score_to_tt(best, ply), depth, flag);
        best
    }

    fn qsearch(&mut self, b: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.check_abort() {
            return 0;
        }
        self.nodes += 1;
        if ply >= MAX_PLY - 1 {
            return evaluate(b);
        }

        let stand_pat = evaluate(b);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut list = MoveList::new();
        generate(b, &mut list, true);
        self.score_moves(b, &mut list, ply, Move::NONE);

        for i in 0..list.len {
            Self::pick_best(&mut list, i);
            let m = list.moves[i];
            if !b.make_move_legal(m) {
                continue;
            }
            let score = -self.qsearch(b, -beta, -alpha, ply + 1);
            b.undo_move();
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }
}
