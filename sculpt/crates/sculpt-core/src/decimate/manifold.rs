//! Is a triangle mesh closed and manifold?

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ManifoldReport {
    /// Edges with one face.
    pub border_edges: usize,
    /// Edges with more than two faces.
    pub nonmanifold_edges: usize,
    /// Vertices whose faces form more than one fan.
    pub nonmanifold_vertices: usize,
    /// Edges two faces use in the same direction (faces that disagree on which side is out).
    pub flipped_edges: usize,
    /// Faces with a repeated vertex.
    pub degenerate_faces: usize,
}

impl ManifoldReport {
    pub fn manifold(&self) -> bool {
        self.nonmanifold_edges == 0 && self.nonmanifold_vertices == 0 && self.degenerate_faces == 0 && self.flipped_edges == 0
    }

    pub fn closed(&self) -> bool {
        self.border_edges == 0
    }
}

pub fn check_manifold(vertex_count: usize, tris: &[[u32; 3]]) -> ManifoldReport {
    let mut r = ManifoldReport { degenerate_faces: tris.par_iter().filter(|t| t[0] == t[1] || t[1] == t[2] || t[2] == t[0]).count(), ..Default::default() };
    // Directed edges as (min, max, forward).
    let mut e: Vec<(u64, bool)> = tris
        .par_iter()
        .flat_map_iter(|t| {
            (0..3).map(move |i| {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                (((a.min(b) as u64) << 32) | a.max(b) as u64, a < b)
            })
        })
        .collect();
    e.par_sort_unstable();
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j].0 == e[i].0 {
            j += 1;
        }
        match j - i {
            1 => r.border_edges += 1,
            2 => {
                if e[i].1 == e[i + 1].1 {
                    r.flipped_edges += 1;
                }
            }
            _ => r.nonmanifold_edges += 1,
        }
        i = j;
    }
    drop(e);
    // Fans: around each vertex, join faces that share an edge at it; more than one group is a pinch.
    let vf = crate::mesh::Csr::from_pairs(vertex_count, tris.iter().enumerate().flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))));
    r.nonmanifold_vertices = (0..vertex_count)
        .into_par_iter()
        .filter(|&v| {
            let fs = vf.row(v);
            if fs.len() < 2 {
                return false;
            }
            // Each face around v contributes the two other vertices as an edge of v's link; count link components.
            let mut links: Vec<(u32, u32)> = fs
                .iter()
                .map(|&f| {
                    let t = tris[f as usize];
                    let k = t.iter().position(|&x| x as usize == v).unwrap();
                    (t[(k + 1) % 3], t[(k + 2) % 3])
                })
                .collect();
            let mut nodes: Vec<u32> = links.iter().flat_map(|&(a, b)| [a, b]).collect();
            nodes.sort_unstable();
            nodes.dedup();
            let mut parent: Vec<usize> = (0..nodes.len()).collect();
            fn find(p: &mut [usize], mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for (a, b) in links.drain(..) {
                let (ia, ib) = (nodes.binary_search(&a).unwrap(), nodes.binary_search(&b).unwrap());
                let (ra, rb) = (find(&mut parent, ia), find(&mut parent, ib));
                parent[ra] = rb;
            }
            let groups = (0..nodes.len()).filter(|&i| find(&mut parent, i) == i).count();
            groups > 1
        })
        .count();
    r
}
