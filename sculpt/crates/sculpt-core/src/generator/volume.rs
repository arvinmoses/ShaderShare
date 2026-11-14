//! The single-resolution volume of version 1 `.svol` files, kept so those files still open (as level 0 of a
//! [`super::level_volume::LevelVolume`]).
//!
//! It is a sparse, clamped distance field: the lattice is cut into blocks of 8x8x8 samples; blocks within a few lattice
//! steps of the surface hold all their samples (distances clamped to +-`band`), every other block holds only whether it
//! is inside or outside. Sculpting happens in `LevelVolume`; this type only reads (and, for tests, writes) the file.

use std::collections::HashMap;

use glam::{IVec3, Vec3};

const N: usize = 512;

/// A sparse, clamped distance field as stored in a version 1 volume file.
pub struct Volume {
    pub voxel: f32,
    pub band: f32,
    origin: Vec3,
    dims: IVec3,
    /// Per block: `-1` inside, `1` outside, `0` stored in full in `dense`.
    signs: Vec<i8>,
    dense: HashMap<u32, Box<[f32; N]>>,
}

impl Volume {
    /// Block coordinates of block `b`.
    pub fn coords_of(&self, b: u32) -> IVec3 {
        let d = self.dims;
        IVec3::new(b as i32 % d.x, (b as i32 / d.x) % d.y, b as i32 / (d.x * d.y))
    }
    pub fn grid(&self) -> (Vec3, IVec3) {
        (self.origin, self.dims)
    }
    /// `-1` inside, `1` outside, `0` stored, per block.
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }
    pub fn stored(&self) -> impl Iterator<Item = (u32, &[f32; N])> {
        self.dense.iter().map(|(b, s)| (*b, &**s))
    }

    /// Write the field to `w`: a small header, the per-block signs run-length coded, and
    /// each stored block's samples at 16 bits.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"SVOL")?;
        w.write_all(&1u32.to_le_bytes())?;
        for f in [self.voxel, self.band, self.origin.x, self.origin.y, self.origin.z] {
            w.write_all(&f.to_le_bytes())?;
        }
        for d in self.dims.to_array() {
            w.write_all(&d.to_le_bytes())?;
        }
        let mut runs: Vec<(i8, u32)> = Vec::new();
        for &s in &self.signs {
            match runs.last_mut() {
                Some((v, n)) if *v == s => *n += 1,
                _ => runs.push((s, 1)),
            }
        }
        w.write_all(&(runs.len() as u32).to_le_bytes())?;
        for (v, n) in runs {
            w.write_all(&[v as u8])?;
            w.write_all(&n.to_le_bytes())?;
        }
        let mut stored: Vec<u32> = self.dense.keys().copied().collect();
        stored.sort_unstable();
        w.write_all(&(stored.len() as u32).to_le_bytes())?;
        let mut buf = Vec::with_capacity(N * 2);
        for b in stored {
            w.write_all(&b.to_le_bytes())?;
            buf.clear();
            for &v in self.dense[&b].iter() {
                // Kept on the same side of zero: the surface must not move by a voxel through rounding.
                let mut q = (v / self.band * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
                if v < 0.0 && q == 0 {
                    q = -1;
                }
                buf.extend_from_slice(&q.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
        Ok(())
    }

    /// Read a field written by [`Self::write_to`].
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Volume> {
        use std::io::{Error, ErrorKind};
        let bad = |m: &str| Error::new(ErrorKind::InvalidData, m.to_string());
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        if &b4 != b"SVOL" {
            return Err(bad("not a volume file"));
        }
        let u32_ = |r: &mut dyn std::io::Read| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        if u32_(r)? != 1 {
            return Err(bad("unknown volume file version"));
        }
        let mut f = [0f32; 5];
        for v in &mut f {
            *v = f32::from_bits(u32_(r)?);
        }
        let mut dims = [0i32; 3];
        for d in &mut dims {
            *d = u32_(r)? as i32;
        }
        let dims = IVec3::from_array(dims);
        let total = (dims.x as usize) * (dims.y as usize) * (dims.z as usize);
        if dims.min_element() <= 0 || total > 1 << 31 {
            return Err(bad("bad lattice size"));
        }
        let mut signs = Vec::with_capacity(total);
        for _ in 0..u32_(r)? {
            let mut v = [0u8; 1];
            r.read_exact(&mut v)?;
            let n = u32_(r)? as usize;
            if signs.len() + n > total {
                return Err(bad("sign runs overflow the lattice"));
            }
            signs.extend(std::iter::repeat_n(v[0] as i8, n));
        }
        if signs.len() != total {
            return Err(bad("sign runs do not cover the lattice"));
        }
        let band = f[1];
        let mut dense = HashMap::new();
        let mut buf = vec![0u8; N * 2];
        for _ in 0..u32_(r)? {
            let b = u32_(r)?;
            if b as usize >= total {
                return Err(bad("block out of range"));
            }
            r.read_exact(&mut buf)?;
            let mut s = Box::new([0f32; N]);
            for (i, v) in s.iter_mut().enumerate() {
                *v = i16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]) as f32 / 32767.0 * band;
            }
            dense.insert(b, s);
        }
        Ok(Volume { voxel: f[0], band, origin: Vec3::new(f[2], f[3], f[4]), dims, signs, dense })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1x1 lattice: one inside block and one stored block holding a ramp.
    fn sample() -> Volume {
        let mut dense = HashMap::new();
        let mut s = Box::new([0f32; N]);
        for (i, x) in s.iter_mut().enumerate() {
            *x = (i as f32 / N as f32 - 0.5) * 0.04;
        }
        dense.insert(1, s);
        Volume { voxel: 0.01, band: 0.04, origin: Vec3::new(-0.08, -0.04, -0.04), dims: IVec3::new(2, 1, 1), signs: vec![-1, 0], dense }
    }

    #[test]
    fn a_volume_survives_saving() {
        let v = sample();
        let mut data = Vec::new();
        v.write_to(&mut data).unwrap();
        let w = Volume::read_from(&mut data.as_slice()).unwrap();
        assert_eq!((w.voxel, w.band, w.grid()), (v.voxel, v.band, v.grid()));
        assert_eq!(w.signs(), v.signs());
        let (a, b): (Vec<_>, Vec<_>) = (v.stored().collect(), w.stored().collect());
        assert_eq!(a.len(), b.len());
        for ((ia, sa), (ib, sb)) in a.into_iter().zip(b) {
            assert_eq!(ia, ib);
            assert!(sa.iter().zip(sb.iter()).all(|(x, y)| (x - y).abs() < 0.04 / 32767.0 + 1e-7));
        }
        assert!(Volume::read_from(&mut &data[..data.len() / 2]).is_err());
    }
}
