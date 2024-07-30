use crate::materials::Material;
use crate::objects::{HitRecord, Hittable, HittableObjects};
use crate::types::ray::Ray;
use core::num;
use std::intrinsics::size_of;
use na::{center, Point3, Vector3};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::cmp::Ordering;
use std::ops::Bound;
use std::sync::{Arc, atomic::AtomicU32, atomic::Ordering::Relaxed};
use typed_arena::Arena;

type Primitive = Arc<dyn Hittable + Send + Sync>;

fn partition<T, F: Fn(&T) -> bool>(items: &mut [T], start: usize, end: usize, cmp: F) -> usize {
    let mut i = start;
    let mut j = end - 1;
    while i < j {
        if !cmp(&items[j]) {
            j -= 1;
            continue;
        }
        if cmp(&items[i]) {
            i += 1;
            continue;
        }
        items.swap(i, j);
    }
    i + 1
}

#[derive(Clone, Copy, Debug)]
pub struct BBox {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl BBox {
    // a, b are oppsite corners
    pub fn new(a: Point3<f32>, b: Point3<f32>) -> Self {
        let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Point3::new(0_f32, 0_f32, 0_f32),
            max: Point3::new(0_f32, 0_f32, 0_f32),
        }
    }

    pub fn centroid(&self) -> Point3<f32> {
        center(&self.min, &self.max)
    }

    pub fn merge(&self, other: &Self) -> Self {
        let min = Point3::new(
            self.min.x.min(other.min.x),
            self.min.y.min(other.min.y),
            self.min.z.min(other.min.z),
        );
        let max = Point3::new(
            self.max.x.max(other.max.x),
            self.max.y.max(other.max.y),
            self.max.z.max(other.max.z),
        );
        Self { min, max }
    }

    pub fn enclose(&self, other: Point3<f32>) -> Self {
        let min = Point3::new(
            self.min.x.min(other.x),
            self.min.y.min(other.y),
            self.min.z.min(other.z),
        );
        let max = Point3::new(
            self.max.x.max(other.x),
            self.max.y.max(other.y),
            self.max.z.max(other.z),
        );
        Self { min, max }
    }

    pub fn get_longest_axis(&self) -> usize {
        let x = self.max.x - self.min.x;
        let y = self.max.y - self.min.y;
        let z = self.max.z - self.min.z;
        if x > y && x > z {
            0
        } else if y > z {
            1
        } else {
            2
        }
    }

    fn get_interval(&self, axis: usize) -> (f32, f32) {
        let min = self.min[axis];
        let max = self.max[axis];
        (min, max)
    }

    pub fn get_surface_area(&self) -> f32 {
        let extent = self.max.coords - self.min.coords;
        2.0 * extent.x * extent.y + 2.0 * extent.x * extent.z + 2.0 * extent.y * extent.z
    }

    pub fn compare(a: &BBox, b: &BBox, axis: usize) -> std::cmp::Ordering {
        let a_bbox_center = a.centroid();
        let b_bbox_center = b.centroid();

        if a_bbox_center[axis] < b_bbox_center[axis] {
            std::cmp::Ordering::Less
        } else if a_bbox_center[axis] > b_bbox_center[axis] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    pub fn offset(&self, p: &Point3<f32>) -> Vector3<f32> {
        let mut o: Vector3<f32> = p - self.min;

        if (self.max.x - self.min.x) != 0.0 {
            o.x /= self.max.x - self.min.x
        }
        if (self.max.y - self.min.y) != 0.0 {
            o.y /= self.max.y - self.min.y
        }
        if (self.max.z - self.min.z) != 0.0 {
            o.z /= self.max.z - self.min.z
        }

        o
    }

    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        let orig = ray.origin;
        let dir = ray.direction;

        let mut t_min: f32 = t_min;
        let mut t_max: f32 = t_max;

        // Find intersection of interval for each axis
        for axis in 0..3 {
            let (min, max) = self.get_interval(axis);
            let adinv: f32 = 1.0_f32 / dir[axis];

            let t0 = (min - orig[axis]) * adinv;
            let t1 = (max - orig[axis]) * adinv;

            if t0 < t1 {
                if t0 > t_min {
                    t_min = t0;
                }
                if t1 < t_max {
                    t_max = t1;
                }
            } else {
                if t1 > t_min {
                    t_min = t1;
                }
                if t0 < t_max {
                    t_max = t0;
                }
            }

            if t_max < t_min {
                return false;
            }
        }
        true
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SplitMethod {
    SAH,
    Middle,
    EqualCounts,
}

#[derive(Copy, Clone, Debug)]
struct SAHBucket {
    count: usize,
    bounds: BBox,
}

// Wrapper class for BVH construction
pub struct BVH {
    max_prims_in_node: usize,
    primitives: Vec<Primitive>,
}

impl BVH {
    pub fn create(
        primitives: Vec<Primitive>,
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> Self {
        let mut bvh = Self {
            primitives: primitives.clone(),
            max_prims_in_node,
        };

        let mut bvh_prims = Vec::new();
        let mut ordered_prims = Vec::new();
        for (i, prim) in primitives.iter().enumerate() {
            bvh_prims.push(BVHPrimitive::new(i, prim.bbox()));
        }
        let len = bvh_prims.len();

        bvh.build_recursive(&mut bvh_prims, &mut ordered_prims, 0, len, split_method);

        bvh
    }

    fn build_recursive(
        &mut self,
        bvh_prims: &mut Vec<BVHPrimitive>,
        ordered_prims: &mut Vec<Primitive>,
        start: usize,
        end: usize,
        split_method: SplitMethod,
    ) -> Box<BVHNode> {
        let bbox = self
            .primitives
            .iter()
            .fold(BBox::empty(), |bbox, prim| bbox.merge(&prim.bbox()));

        let num_prims = end - start;
        match num_prims {
            // LEAF
            1 => {
                // We just push the prims to the ordered list
                let first_prim_offset = ordered_prims.len();
                for i in start..end {
                    let prim_num = bvh_prims[i].index;
                    ordered_prims.push(self.primitives[prim_num].clone());
                }

                // Return a leaf node
                Box::new(BVHNode::new_leaf(first_prim_offset, num_prims, bbox))
            }
            // INTERIOR
            _ => {
                let centroid_bbox = bvh_prims[start..end]
                    .into_iter()
                    .fold(BBox::empty(), |bbox, prim| bbox.enclose(prim.centroid()));

                let split_axis = centroid_bbox.get_longest_axis();
                match centroid_bbox.min[split_axis] == centroid_bbox.max[split_axis] {
                    // LEAF (because range is 0)
                    true => {
                        let first_prim_offset = ordered_prims.len();
                        for i in start..end {
                            let prim_num = bvh_prims[i].index;
                            ordered_prims.push(self.primitives[prim_num].clone());
                        }

                        Box::new(BVHNode::new_leaf(first_prim_offset, num_prims, bbox))
                    }
                    false => {
                        // Recurse
                        let mid = match split_method {
                            // TODO: Implement other methods
                            SplitMethod::Middle => {
                                // Middle
                                let axis_center = centroid_bbox.min[split_axis]
                                    + (centroid_bbox.max[split_axis]
                                        - centroid_bbox.min[split_axis])
                                        / 2.0;
                                let mid = partition(bvh_prims, start, end, |prim| {
                                    prim.centroid()[split_axis] < axis_center
                                });

                                if mid == start || mid == end {
                                    Some(start + (end - start) / 2)
                                } else {
                                    Some(mid)
                                }
                            }
                            _ => {
                                match num_prims <= 4 {
                                    true => {
                                        // sort list slice
                                        bvh_prims[start..end].sort_by(|a, b| {
                                            a.centroid()[split_axis]
                                                .partial_cmp(&b.centroid()[split_axis])
                                                .unwrap_or(Ordering::Less)
                                        });
                                        Some(start + (end - start) / 2)
                                    }
                                    false => {
                                        const NUM_BUCKETS: usize = 12;
                                        let mut buckets = vec![
                                            SAHBucket {
                                                count: 0,
                                                bounds: BBox::empty()
                                            };
                                            NUM_BUCKETS
                                        ];

                                        for i in start..end {
                                            let prim = &bvh_prims[i];
                                            let mut b = (prim.centroid()[split_axis]
                                                * NUM_BUCKETS as f32)
                                                as usize;
                                            if b == NUM_BUCKETS {
                                                b = NUM_BUCKETS - 1;
                                            }
                                            buckets[b].count += 1;
                                            buckets[b].bounds =
                                                buckets[b].bounds.merge(&prim.bbox());
                                        }

                                        let mut cost = [0 as f32; NUM_BUCKETS - 1];
                                        for i in 0..(NUM_BUCKETS - 1) {
                                            let mut b0 = BBox::empty();
                                            let mut b1 = BBox::empty();

                                            let mut count0 = 0;
                                            let mut count1 = 0;
                                            for j in 0..i {
                                                b0 = b0.merge(&buckets[j].bounds);
                                                count0 += buckets[j].count;
                                            }
                                            for j in i + 1..NUM_BUCKETS {
                                                b1 = b1.merge(&buckets[j].bounds);
                                                count1 += buckets[j].count;
                                            }
                                            cost[i] = 0.125
                                                + (count0 as f32 * b0.get_surface_area()
                                                    + count1 as f32 * b1.get_surface_area())
                                                    / bbox.get_surface_area();
                                        }

                                        let (min_cost_bucket, _) = cost
                                            .iter()
                                            .enumerate()
                                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                            .unwrap();
                                        let min_cost = cost[min_cost_bucket];

                                        let leaf_cost = num_prims as f32;
                                        if num_prims > self.max_prims_in_node
                                            || min_cost < leaf_cost
                                        {
                                            Some(partition(bvh_prims, start, end, |prim| {
                                                let b = NUM_BUCKETS as f32 * centroid_bbox.offset(&prim.centroid())[split_axis];
                                                b.min((NUM_BUCKETS - 1) as f32) <= min_cost_bucket as f32
                                            }))
                                        } else {
                                            None
                                        }
                                    }
                                }
                            }
                        };

                        // Create BVH nodes
                        match mid {
                            Some(mid) => {
                                let left = self.build_recursive(
                                    bvh_prims,
                                    ordered_prims,
                                    start,
                                    mid,
                                    split_method,
                                );
                                let right = self.build_recursive(
                                    bvh_prims,
                                    ordered_prims,
                                    mid,
                                    end,
                                    split_method,
                                );
                                Box::new(BVHNode::new_interior(left, right, split_axis))
                            }
                            None => {
                                let first_prim_offset = ordered_prims.len();
                                for i in start..end {
                                    let prim_num = bvh_prims[i].index;
                                    ordered_prims.push(self.primitives[prim_num].clone());
                                }
                                Box::new(BVHNode::new_leaf(first_prim_offset, num_prims, bbox))
                            }
                        }
                    }
                }
            }
        }
    }

    fn hlbvh_build(
        &mut self,
        bvh_prims: &mut Vec<BVHPrimitive>,
        ordered_prims: &mut Vec<Primitive>,
    ) -> Box<BVHNode> {
        let centroid_bbox = bvh_prims.iter().fold(BBox::empty(), |bbox, prim| {
            bbox.enclose(prim.centroid())
        });

        let mut morton_prims: Vec<MortonPrimitive> = bvh_prims.par_iter().enumerate().map(|(i, prim)| {
            const MORTON_BITS: usize = 10;
            const MORTON_SCALE: usize = 1 << MORTON_BITS;

            let centroid_offset = centroid_bbox.offset(&prim.centroid());
            
            MortonPrimitive::new(
                prim.index,
                MortonPrimitive::encode(&(centroid_offset * MORTON_SCALE as f32)) 
            )
        }).collect();

        MortonPrimitive::radix_sort(&mut morton_prims);

        let mut treelets_to_build: Vec<LBVHTreelet> = vec![];
        let mut start = 0;
        for end in 1..morton_prims.len()-1 {
            const MASK: u32 = 0b00111111111111000000000000000000;
            if end == morton_prims.len() || ((morton_prims[start].morton_code & MASK) != (morton_prims[end].morton_code & MASK)) {
                let num_prims = end - start;
                let max_bvh_nodes = 2 * num_prims -1;

                let nodes = (0..max_bvh_nodes).map(|_| {
                    Box::new(BVHNode::new_leaf(0, 0, BBox::empty()))
                }).collect();

                let mut v = Vec::with_capacity(max_bvh_nodes * std::mem::size_of::<Box<BVHNode>>());


                treelets_to_build.push(LBVHTreelet {
                    start,
                    size: num_prims,
                    build_nodes: vec![&Box::new(BVHNode::new_leaf(0, 0, BBox::empty())); max_bvh_nodes]
                });

                start = end;
            }
        }

        let mut ordered_prims_offset: AtomicU32 = AtomicU32::new(0);
        treelets_to_build.par_iter().for_each(|treelet| {
            let mut nodes_created = treelet.build_nodes.clone();
            const FIRST_BIT_INDEX: i32 = 29 - 12;
            treelet.build_nodes = 
        })

        

        Box::new(BVHNode::new_leaf(0, 0, BBox::empty()))
    }

    fn emit_lbvh(
        &self,
        to_build: &mut usize,
        bvh_prims: &Vec<BVHPrimitive>,
        morton_prims: &Vec<MortonPrimitive>,
        num_prims: usize,
        total_nodes: &mut usize,
        ordered_prims: &mut Vec<Primitive>,
        ordered_prims_offset: &mut AtomicU32,
        bit_index: i32,
    ) -> Vec<Box<BVHNode>> {
        if bit_index == -1 || num_prims < self.max_prims_in_node {
            *total_nodes += 1;
            *to_build -= 1;
            let mut bbox = BBox::empty();
            let first_prim_offset = ordered_prims_offset.fetch_add(num_prims as u32, Relaxed) as usize;

            for i in 0..num_prims {
                let prim_index = bvh_prims[i].index;
                ordered_prims.push(self.primitives[prim_index].clone());
                bbox = bbox.merge(&self.primitives[prim_index].bbox());
            }

            vec![Box::new(BVHNode::new_leaf(first_prim_offset, num_prims, bbox))]
        }
        else {
            let mask = 1 << bit_index;

            if (morton_prims[0].morton_code & mask) == (morton_prims[num_prims - 1].morton_code & mask) { 
                let new = self.emit_lbvh(to_build, bvh_prims, morton_prims, num_prims, total_nodes, ordered_prims, ordered_prims_offset, bit_index)

                new
            } else {
                let mut search_start = 0;
                let mut search_end = num_prims - 1;

                while search_start + 1 != search_end {
                    let mid = search_start + (search_end - search_start) / 2;

                    if (morton_prims[search_start].morton_code & mask) == (morton_prims[mid].morton_code & mask) {
                        search_start = mid;
                    } else {
                        search_end = mid;
                    }
                }

                let split_offset = search_end;

                *total_nodes += 1;
                *to_build -= 1;

                let mut left = self.emit_lbvh(to_build, bvh_prims, morton_prims, split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1);
                let mut right = self.emit_lbvh(to_build, bvh_prims, morton_prims, num_prims - split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1);

                let axis = bit_index % 3;

                BVHNode::new_interior(left, right, split_axis)
            }

        }
    }
}

impl Hittable for BVH {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        None
    }
    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        None
    }
    fn bbox(&self) -> BBox {
        BBox::empty()
    }
}

struct LBVHTreelet {
    start: usize,
    size: usize,
    build_nodes: Vec<Box<BVHNode>>,
}

#[derive(Copy, Clone, Debug)]
struct MortonPrimitive {
    index: usize,
    morton_code: u32
}

impl MortonPrimitive {
    pub fn new(index: usize, morton_code: u32) -> Self {
        Self { index, morton_code }
    }

    pub fn set(&mut self, index: usize, morton_code: u32) {
        self.index = index;
        self.morton_code = morton_code;
    }

    pub fn encode(v: &Vector3<f32>) -> u32 {
        Self::lshift3(v.z.to_bits()) << 2 | Self::lshift3(v.y.to_bits()) << 1 | Self::lshift3(v.x.to_bits())
    }

    fn lshift3(x: u32) -> u32 {
        let x = if x == (1 << 10) {x-1} else {x};
        let x = (x | (x << 16)) & 0b00000011000000000000000011111111;
        let x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
        let x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
        let x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
        return x;
    }

    pub fn radix_sort(prims: &mut Vec<MortonPrimitive>) {
        let mut temp = vec![MortonPrimitive::new(0, 0); prims.len()];
        const BITS_PER_PASS: usize = 5;
        const N_BITS: usize = 30;
        const N_PASSES: usize = N_BITS / BITS_PER_PASS;

        for pass in 0..N_PASSES {
            let low_bit = pass * BITS_PER_PASS;

            let (inn, out) = match pass & 1 {
                0 => (&mut *prims, &mut temp),
                _ => (&mut temp, &mut *prims),
            };

            const N_BUCKETS: usize = 1 << BITS_PER_PASS;
            const BITMASK: u32 = (1 << BITS_PER_PASS) - 1;
            let mut bucket_count = vec![0; N_BUCKETS];

            for prim in inn.iter() {
                let bucket = ((prim.morton_code >> low_bit) & BITMASK) as usize;
                bucket_count[bucket] += 1;
            }

            let mut out_index = vec![0; N_BUCKETS];
            out_index[0] = 0;
            for i in 1..N_BUCKETS {
                out_index[i] = out_index[i - 1] + bucket_count[i - 1];
            }

            for prim in inn.iter() {
                let bucket = ((prim.morton_code >> low_bit) & BITMASK) as usize;
                out[out_index[bucket]] = *prim;
                out_index[bucket] += 1;
            }
        }

        if (N_PASSES & 1) == 1 {
            std::mem::swap(prims, &mut temp);
        }
    }
}

// Virtual primitive
struct BVHPrimitive {
    index: usize,
    bbox: BBox,
}

impl BVHPrimitive {
    pub fn new(index: usize, bbox: BBox) -> Self {
        Self { index, bbox }
    }

    pub fn bbox(&self) -> BBox {
        self.bbox
    }

    pub fn centroid(&self) -> Point3<f32> {
        self.bbox.centroid()
    }
}

// BVH Node
struct BVHNode {
    // For interior nodes
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    split_axis: Option<usize>,

    // For leaf nodes
    // offset, size
    bounds: Option<(usize, usize)>,

    // BBox
    bbox: BBox,
}

impl BVHNode {
    pub fn new_leaf(offset: usize, size: usize, bbox: BBox) -> Self {
        Self {
            left: None,
            right: None,
            split_axis: None,
            bounds: Some((offset, size)),
            bbox,
        }
    }

    pub fn new_interior(left: Box<BVHNode>, right: Box<BVHNode>, split_axis: usize) -> Self {
        let bbox = left.bbox.merge(&right.bbox);
        Self {
            left: Some(left),
            right: Some(right),
            split_axis: Some(split_axis),
            bounds: None,
            bbox,
        }
    }
}

// impl Hittable for BVHNode {
//     fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
//         if !self.bbox.hit(ray, t_min, t_max) {
//             return None;
//         }

//         let hit_left = self.left.hit(ray, t_min, t_max);
//         let hit_right = self.right.hit(ray, t_min, t_max);

//         match (hit_left, hit_right) {
//             (Some(hit_left), Some(hit_right)) => {
//                 if hit_left.t() < hit_right.t() {
//                     Some(hit_left)
//                 } else {
//                     Some(hit_right)
//                 }
//             }
//             (Some(hit_left), None) => Some(hit_left),
//             (None, Some(hit_right)) => Some(hit_right),
//             (None, None) => None,
//         }
//     }
//     fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
//         None
//     }
//     fn bbox(&self) -> BBox {
//         self.bbox
//     }
// }
