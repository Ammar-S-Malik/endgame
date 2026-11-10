//! Transposition table: a hash map from position (Zobrist key) to the result of a
//! previous search of that position (score, bound type, depth, best move).
//!
//! Each slot is 16 bytes. The table size is rounded down to a power of two so the
//! index is a mask of the key. Replacement prefers deeper and more recent entries.

use crate::types::Move;

pub const FLAG_EXACT: u8 = 0;
/// The true score is at least `score` (the search failed high).
pub const FLAG_LOWER: u8 = 1;
/// The true score is at most `score` (the search failed low).
pub const FLAG_UPPER: u8 = 2;

#[derive(Clone, Copy, Default)]
pub struct TTEntry {
    pub key: u64,
    pub mv: u16,
    pub score: i16,
    pub depth: i8,
    pub flag: u8,
    pub age: u8,
}

pub struct TranspositionTable {
    table: Vec<TTEntry>,
    mask: usize,
    age: u8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> TranspositionTable {
        let mut t = TranspositionTable { table: Vec::new(), mask: 0, age: 0 };
        t.resize(mb);
        t
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / std::mem::size_of::<TTEntry>();
        if !n.is_power_of_two() {
            n = n.next_power_of_two() / 2;
        }
        self.table = vec![TTEntry::default(); n];
        self.mask = n - 1;
        self.age = 0;
    }

    pub fn clear(&mut self) {
        for e in self.table.iter_mut() {
            *e = TTEntry::default();
        }
        self.age = 0;
    }

    /// Call at the start of every search so old entries become replaceable.
    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let e = self.table[(key as usize) & self.mask];
        if e.key == key && key != 0 {
            Some(e)
        } else {
            None
        }
    }

    #[inline]
    pub fn store(&mut self, key: u64, mv: Move, score: i32, depth: i32, flag: u8) {
        let idx = (key as usize) & self.mask;
        let age = self.age;
        let e = &mut self.table[idx];
        let depth = depth.clamp(-127, 127);
        if e.key == key {
            // Same position: overwrite unless the old entry is deeper (and fresh).
            if depth >= e.depth as i32 || flag == FLAG_EXACT || e.age != age {
                let keep_move = if mv == Move::NONE { e.mv } else { mv.0 };
                *e = TTEntry { key, mv: keep_move, score: score as i16, depth: depth as i8, flag, age };
            } else if mv != Move::NONE && e.mv == 0 {
                e.mv = mv.0;
            }
        } else if e.key == 0 || e.age != age || depth + 2 >= e.depth as i32 {
            *e = TTEntry { key, mv: mv.0, score: score as i16, depth: depth as i8, flag, age };
        }
    }
}
