//! NNUE: an efficiently updatable neural-network evaluation.
//!
//! Architecture: (768 x KING_BUCKETS) inputs -> H hidden (per perspective, shared weights) ->
//! clipped ReLU -> concat(side to move, other side) -> 1 output. The output layer has
//! OUT_BUCKETS separate sets of weights, chosen by the number of pieces on the board.
//!
//! * Input feature = (piece colour relative to the perspective, piece type, square), where
//!   the black perspective mirrors the board vertically. 2 * 6 * 64 = 768 features per bucket.
//! * King buckets: the feature block used for a perspective depends on where that
//!   perspective's own king stands (see `king_bucket`), so the network can learn different
//!   weights for a king on its back rank, in the centre, and so on.
//! * The hidden layer ("accumulator") is a sum of the weight rows of the active features, so
//!   moving a piece only adds and subtracts two rows instead of recomputing everything. When a
//!   king moves into a different bucket, that perspective's accumulator is rebuilt from scratch.
//! * Weights are quantised to i16: feature weights and biases scaled by QA, output weights
//!   scaled by QB, so inference is integer-only.
//!
//! File format (little endian): magic "ENN2", u32 H, u32 KING_BUCKETS, u32 OUT_BUCKETS, then
//! i16 ft_w[KING_BUCKETS*768*H], i16 ft_b[H], i16 out_w[OUT_BUCKETS*2*H], i32 out_b[OUT_BUCKETS].
//! Output bucket `k` uses out_w[k*2*H ..][..2*H] and out_b[k].

use crate::board::Board;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Hidden layer size. The trainer (`trainer/train.py --hidden`) must use the same value.
pub const H: usize = 64;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
/// Network output of 1.0 corresponds to this many centipawns.
pub const SCALE: i32 = 400;
/// Features in one king bucket (2 colours * 6 piece types * 64 squares).
pub const BUCKET_INPUTS: usize = 768;
pub const KING_BUCKETS: usize = 8;
pub const INPUTS: usize = BUCKET_INPUTS * KING_BUCKETS;
pub const OUT_BUCKETS: usize = 8;
pub const MAGIC: &[u8; 4] = b"ENN2";
const HEADER: usize = 16;

pub type Accumulators = [[i16; H]; 2];

pub struct Net {
    /// `ft_w[feature * H + i]`, `INPUTS * H` values.
    pub ft_w: Vec<i16>,
    pub ft_b: [i16; H],
    /// `out_w[bucket * 2 * H + i]`; the first `H` weights multiply the side to move's hidden
    /// layer, the next `H` the other side's.
    pub out_w: Vec<i16>,
    pub out_b: [i32; OUT_BUCKETS],
}

/// The active network (null = none, use the handcrafted evaluation). Networks are installed
/// by leaking a `Box`, so a pointer read here stays valid forever; swapping a network leaks
/// the old one (under 1 MB), which only happens on a UCI option change.
static NET: AtomicPtr<Net> = AtomicPtr::new(std::ptr::null_mut());

/// The default network, embedded in the binary at build time.
pub static DEFAULT_NET: &[u8] = include_bytes!("../net/endgame-v2.nnue");

/// Size in bytes of a network file.
pub const fn file_size() -> usize {
    HEADER + 2 * (INPUTS * H + H + OUT_BUCKETS * 2 * H) + 4 * OUT_BUCKETS
}

/// The active network, if any. When `None`, the engine uses its handcrafted evaluation.
#[inline]
pub fn net() -> Option<&'static Net> {
    // SAFETY: the pointer is either null or comes from `Box::leak`, so it is never freed.
    unsafe { NET.load(Ordering::Relaxed).as_ref() }
}

fn install(net: Net) {
    NET.store(Box::leak(Box::new(net)), Ordering::Relaxed);
}

/// Switch to the handcrafted evaluation.
pub fn disable() {
    NET.store(std::ptr::null_mut(), Ordering::Relaxed);
}

/// Install the network embedded in the binary.
pub fn load_default() -> Result<(), String> {
    load_bytes(DEFAULT_NET)
}

/// King bucket for a perspective whose own king stands on `ksq` (a real board square).
/// The king's square is seen from the perspective's side (rank 0 = its own back rank):
/// rank groups {0}, {1}, {2,3}, {4..7} times file halves {a-d}, {e-h} give 8 buckets.
#[inline]
pub fn king_bucket(perspective: usize, ksq: usize) -> usize {
    let rel = if perspective == 0 { ksq } else { ksq ^ 56 };
    let rank = rel / 8;
    let file = rel % 8;
    let rank_group = match rank {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    rank_group * 2 + (file >= 4) as usize
}

/// Number of the output head used for a position with `piece_count` pieces (kings included).
#[inline]
pub fn output_bucket(piece_count: u32) -> usize {
    (((piece_count.max(2) - 2) / 4) as usize).min(OUT_BUCKETS - 1)
}

/// Feature index of a piece (`code` = colour * 6 + type) on `sq`, seen from `perspective`,
/// whose own king stands on `ksq`.
#[inline]
pub fn feature(perspective: usize, code: usize, sq: usize, ksq: usize) -> usize {
    let color = code / 6;
    let pt = code % 6;
    let base = if perspective == 0 {
        (color * 6 + pt) * 64 + sq
    } else {
        ((color ^ 1) * 6 + pt) * 64 + (sq ^ 56)
    };
    king_bucket(perspective, ksq) * BUCKET_INPUTS + base
}

impl Net {
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, String> {
        if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
            return Err(
                "not an Endgame NNUE v2 file (convert an older network with tools/convert_net.py)".to_string(),
            );
        }
        let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let (h, kb, ob) = (rd(4), rd(8), rd(12));
        if h != H || kb != KING_BUCKETS || ob != OUT_BUCKETS {
            return Err(format!(
                "network is {h}/{kb}/{ob} (hidden/king buckets/output buckets), engine is built for {H}/{KING_BUCKETS}/{OUT_BUCKETS}"
            ));
        }
        if bytes.len() != file_size() {
            return Err(format!("bad file size {} (expected {})", bytes.len(), file_size()));
        }
        let mut pos = HEADER;
        let mut read_i16 = |n: usize| -> Vec<i16> {
            let v: Vec<i16> = bytes[pos..pos + 2 * n]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            pos += 2 * n;
            v
        };
        let ft_w = read_i16(INPUTS * H);
        let ft_b_v = read_i16(H);
        let out_w = read_i16(OUT_BUCKETS * 2 * H);
        let mut ft_b = [0i16; H];
        ft_b.copy_from_slice(&ft_b_v);
        let mut out_b = [0i32; OUT_BUCKETS];
        for k in 0..OUT_BUCKETS {
            out_b[k] = i32::from_le_bytes(bytes[pos + 4 * k..pos + 4 * k + 4].try_into().unwrap());
        }
        Ok(Net { ft_w, ft_b, out_w, out_b })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(file_size());
        v.extend_from_slice(MAGIC);
        for x in [H as u32, KING_BUCKETS as u32, OUT_BUCKETS as u32] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in self.ft_w.iter().chain(self.ft_b.iter()).chain(self.out_w.iter()) {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in self.out_b.iter() {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    /// Add (`add` = true) or remove (`add` = false) a piece in one perspective's accumulator;
    /// `ksq` is that perspective's own king square.
    #[inline]
    pub fn update(&self, acc: &mut Accumulators, p: usize, code: usize, sq: usize, ksq: usize, add: bool) {
        let row = &self.ft_w[feature(p, code, sq, ksq) * H..][..H];
        let a = &mut acc[p];
        if add {
            for i in 0..H {
                a[i] = a[i].wrapping_add(row[i]);
            }
        } else {
            for i in 0..H {
                a[i] = a[i].wrapping_sub(row[i]);
            }
        }
    }

    /// Rebuild one perspective's accumulator from scratch from a board's pieces.
    pub fn refresh_perspective(&self, b: &Board, p: usize, acc: &mut Accumulators) {
        let ksq = b.king_sq(p);
        acc[p] = self.ft_b;
        for sq in 0..64 {
            let code = b.mailbox[sq] as usize;
            if code < 12 {
                self.update(acc, p, code, sq, ksq, true);
            }
        }
    }

    /// Rebuild both accumulators from scratch.
    pub fn refresh(&self, b: &Board) -> Accumulators {
        let mut acc = [self.ft_b; 2];
        self.refresh_perspective(b, 0, &mut acc);
        self.refresh_perspective(b, 1, &mut acc);
        acc
    }

    /// Bring the accumulators up to date after a move. `b` is the board after the move,
    /// `changes` lists every piece the move added or removed as (piece code, square, added),
    /// and `old_ksq` holds both kings' squares before the move. A perspective whose king moved
    /// into another bucket is rebuilt; the other updates are incremental.
    pub fn apply(&self, acc: &mut Accumulators, b: &Board, changes: &[(usize, usize, bool)], old_ksq: [usize; 2]) {
        for p in 0..2 {
            let ksq = b.king_sq(p);
            if king_bucket(p, ksq) != king_bucket(p, old_ksq[p]) {
                self.refresh_perspective(b, p, acc);
            } else {
                for &(code, sq, add) in changes {
                    self.update(acc, p, code, sq, ksq, add);
                }
            }
        }
    }

    /// Evaluate from the side to move's point of view, in centipawns. `piece_count` is the
    /// number of pieces on the board (kings included); it selects the output head.
    #[inline]
    pub fn evaluate(&self, acc: &Accumulators, side: usize, piece_count: u32) -> i32 {
        let us = &acc[side];
        let them = &acc[side ^ 1];
        let bucket = output_bucket(piece_count);
        let w = &self.out_w[bucket * 2 * H..][..2 * H];
        let mut sum: i32 = self.out_b[bucket];
        for i in 0..H {
            sum += (us[i].clamp(0, QA as i16) as i32) * (w[i] as i32);
        }
        for i in 0..H {
            sum += (them[i].clamp(0, QA as i16) as i32) * (w[H + i] as i32);
        }
        sum * SCALE / (QA * QB)
    }
}

/// Load a network from a file and make it the engine's evaluation. Boards created or
/// refreshed afterwards pick it up (see `Board::refresh_nnue`).
pub fn load_file(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    load_bytes(&bytes)
}

/// Install a network from raw bytes.
pub fn load_bytes(bytes: &[u8]) -> Result<(), String> {
    install(Net::from_bytes(bytes)?);
    Ok(())
}
