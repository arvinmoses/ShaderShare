//! Square grids of heights over a brush's disc, shared by the volume's smooth (`generator::level_volume`) and the mesh
//! layers' (`brush::FormSmooth`): the blur, and the quartic the smooth keeps the form with.

use rayon::prelude::*;

/// The heights `h` (a `size` by `size` grid, row by row) smoothed over `radius` columns either way (a box three times
/// over, so near a Gaussian). A column with no height (NaN) takes its neighbours' mean, when any has one.
pub(crate) fn blur(h: &mut Vec<f32>, size: i32, radius: i32) {
    // A row at a time over the threads for a wide brush's (a few hundred columns each way); a small grid's on this
    // thread, where handing out its rows would cost more than they do. The same sums either way.
    let parallel = size * size * (2 * radius + 1) >= 1 << 16;
    let mut prev = h.clone();
    for _ in 0..3 {
        for axis in 0..2 {
            prev.copy_from_slice(h);
            let prev = &prev;
            let row_of = |(b, row): (usize, &mut [f32])| {
                let (b, n) = (b as i32, size as usize);
                for a in 0..size {
                    // The columns within `radius` along `axis` that are in the grid, in order.
                    let (along, first) = if axis == 0 { (a, b * size) } else { (b, a) };
                    let (lo, hi) = ((along - radius).max(0), (along + radius).min(size - 1));
                    let (mut sum, mut count) = (0.0f32, 0usize);
                    let mut add = |v: f32| {
                        if v.is_finite() {
                            sum += v;
                            count += 1;
                        }
                    };
                    if axis == 0 {
                        prev[(first + lo) as usize..=(first + hi) as usize].iter().for_each(|&v| add(v));
                    } else {
                        prev[(lo * size + first) as usize..=(hi * size + first) as usize].iter().step_by(n).for_each(|&v| add(v));
                    }
                    row[a as usize] = if count > 0 { sum / count as f32 } else { f32::NAN };
                }
            };
            if parallel {
                h.par_chunks_mut(size as usize).enumerate().for_each(row_of);
            } else {
                h.chunks_mut(size as usize).enumerate().for_each(row_of);
            }
        }
    }
}

/// The terms of a polynomial in `a`, `b` up to the fourth degree.
pub(crate) fn quartic_terms(a: f64, b: f64) -> [f64; 15] {
    let (a2, b2) = (a * a, b * b);
    [1.0, a, b, a2, a * b, b2, a2 * a, a2 * b, a * b2, b2 * b, a2 * a2, a2 * a * b, a2 * b2, a * b2 * b, b2 * b2]
}

/// `m x = v` by elimination with partial pivoting; `None` when `m` is (near) singular.
pub(crate) fn solve<const K: usize>(mut m: [[f64; K]; K], mut v: [f64; K]) -> Option<[f64; K]> {
    let scale = (0..K).map(|i| m[i][i].abs()).fold(0.0, f64::max).max(1e-300);
    for c in 0..K {
        let p = (c..K).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        if m[p][c].abs() < 1e-12 * scale {
            return None;
        }
        m.swap(c, p);
        v.swap(c, p);
        for r in c + 1..K {
            let f = m[r][c] / m[c][c];
            for k in c..K {
                m[r][k] -= f * m[c][k];
            }
            v[r] -= f * v[c];
        }
    }
    let mut x = [0.0f64; K];
    for c in (0..K).rev() {
        let s: f64 = (c + 1..K).map(|k| m[c][k] * x[k]).sum();
        x[c] = (v[c] - s) / m[c][c];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blur as the volume's smooth had it, column by column with every bound checked.
    fn blur_as_it_was(h: &mut Vec<f32>, size: i32, radius: i32) {
        for _ in 0..3 {
            for axis in 0..2 {
                let prev = h.clone();
                for b in 0..size {
                    for a in 0..size {
                        let (mut sum, mut count) = (0.0f32, 0usize);
                        for d in -radius..=radius {
                            let (x, y) = if axis == 0 { (a + d, b) } else { (a, b + d) };
                            if x < 0 || y < 0 || x >= size || y >= size {
                                continue;
                            }
                            let v = prev[(y * size + x) as usize];
                            if v.is_finite() {
                                sum += v;
                                count += 1;
                            }
                        }
                        h[(b * size + a) as usize] = if count > 0 { sum / count as f32 } else { f32::NAN };
                    }
                }
            }
        }
    }

    #[test]
    fn blur_is_bit_for_bit_what_it_was() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for (size, radius) in [(5, 1), (43, 3), (31, 2), (121, 9), (301, 20)] {
            let h: Vec<f32> = (0..size * size).map(|_| if rnd() < 0.1 { f32::NAN } else { rnd() * 2.0 - 1.0 }).collect();
            let (mut a, mut b) = (h.clone(), h);
            blur(&mut a, size, radius);
            blur_as_it_was(&mut b, size, radius);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "size {size} radius {radius}");
        }
    }
}
