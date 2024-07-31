use crate::materials::Material;
use crate::objects::{HitRecord, Hittable, HittableObjects, Primitive};
use crate::types::ray::Ray;
use na::{center, Point3};
use std::sync::Arc;

type BVHNodePtr = Box<BVHNode>;

#[derive(Clone, Copy, Debug)]
pub struct BBox {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl BBox {
    // a, b are oppsite corners
    #[inline(always)]
    pub fn new(a: Point3<f32>, b: Point3<f32>) -> Self {
        let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        Self { min, max }
    }

    #[inline(always)]
    pub fn empty() -> Self {
        Self {
            min: Point3::new(0_f32, 0_f32, 0_f32),
            max: Point3::new(0_f32, 0_f32, 0_f32),
        }
    }

    #[inline(always)]
    pub fn point(point: Point3<f32>) -> Self {
        Self {
            min: point,
            max: point
        }
    }

    #[inline(always)]
    pub fn centroid(&self) -> Point3<f32> {
        center(&self.min, &self.max)
    }

    #[inline(always)]
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

    #[inline(always)]
    pub fn enclose(&self, other: &Point3<f32>) -> Self {
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

    #[inline(always)]
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

    #[inline(always)]
    fn get_interval(&self, axis: usize) -> (f32, f32) {
        let min = self.min[axis];
        let max = self.max[axis];
        (min, max)
    }

    #[inline(always)]
    pub fn get_surface_area(&self) -> f32 {
        let extent = self.max.coords - self.min.coords;
        2.0 * extent.x * extent.y + 2.0 * extent.x * extent.z + 2.0 * extent.y * extent.z
    }

    pub fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(f32, f32)> {
        let mut t_min: f32 = t_min;
        let mut t_max: f32 = t_max;

        // Find intersection of interval for each axis
        for axis in 0..3 {
            // Get bounds for the current axis
            let (min, max) = self.get_interval(axis);

            // Calculate intersection times of the ray with the interval
            let adinv: f32 = 1.0 / &ray.direction[axis];
            let t0 = (min - &ray.origin[axis]) * adinv;
            let t1 = (max - &ray.origin[axis]) * adinv;

            // Shrink the interval (t_min, t_max) to the intersection interval
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
                return None;
            }
        }

        Some((t_min, t_max))
    }

    pub fn intersect_bool(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        let mut t_min: f32 = t_min;
        let mut t_max: f32 = t_max;

        // Find intersection of interval for each axis
        for axis in 0..3 {
            // Get bounds for the current axis
            let (min, max) = self.get_interval(axis);

            // Calculate intersection times of the ray with the interval
            let adinv: f32 = 1.0 / &ray.direction[axis];
            let t0 = (min - &ray.origin[axis]) * adinv;
            let t1 = (max - &ray.origin[axis]) * adinv;

            // Shrink the interval (t_min, t_max) to the intersection interval
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

    fn bbox_comp(a: &BBox, b: &BBox, axis: usize) -> std::cmp::Ordering {
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

    #[inline(always)]
    fn longest_axis(&self) -> usize {
        let extent = self.max - self.min;
        if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        }
    }
}

pub enum SplitMethod {
    SAH,
    EqualCounts,
    Middle,
}

pub struct BVHBuilder {
    split_method: SplitMethod,
    max_leaf_size: usize,
}

impl BVHBuilder {
    pub fn build_from_hittable_objects(split_method: SplitMethod, objects: HittableObjects) -> BVH {
        let builder = BVHBuilder {
            split_method,
            max_leaf_size: 10,
        };

        // We clone objects to avoid mutating the original
        let mut objects = objects.objs_clone();
        let len = objects.len();

        BVH {
            root: builder.build_bvh(&mut objects, 0, len),
            ordered_objects: objects,
        }
    }

    pub fn build_flattened_from_hittable_objects(
        split_method: SplitMethod,
        objects: HittableObjects,
    ) -> FlatBVH {
        let builder = BVHBuilder {
            split_method,
            max_leaf_size: 3,
        };

        // We clone objects to avoid mutating the original
        let mut objects = objects.objs_clone();
        let len = objects.len();

        let root = builder.build_bvh(&mut objects, 0, len);
        let nodes = builder.flatten_bvh(&root);

        let nodes = nodes.into();

        FlatBVH {
            nodes,
            ordered_objects: objects,
        }
    }

    fn build_bvh(&self, objects: &mut Vec<Primitive>, start: usize, end: usize) -> BVHNodePtr {
        assert!(start < end, "{} {}", start, end);
        // Compute bounds of all objects in the list
        let bbox = objects[start..end]
            .iter()
            .fold(objects[start].bbox(), |acc, obj| acc.merge(&obj.bbox()));

        let axis = bbox.longest_axis();
        let size = end - start;
        // TODO: Set max leaf size.
        match size <= self.max_leaf_size {
            true => {
                BVHNode::new_leaf_as_box(start, size, bbox)
            } 
            false => {
                let centroid_bbox = objects[start..end].iter().fold(
                    BBox::point(objects[start].bbox().centroid()),
                    |acc, obj| acc.enclose(&obj.bbox().centroid()),
                );

                // Return leaf if the bounding boxes are too small
                if centroid_bbox.max[axis] - centroid_bbox.min[axis] <= 0.0001 {
                    return BVHNode::new_leaf_as_box(start, size, bbox);
                }

                let mid = match self.split_method {
                    SplitMethod::EqualCounts => {
                        objects[start..end].sort_by(|a, b| BBox::bbox_comp(&a.bbox(), &b.bbox(), axis));

                        start + size / 2
                    }
                    _ => {
                        objects[start..end].sort_by(|a, b| BBox::bbox_comp(&a.bbox(), &b.bbox(), axis));

                        let center = (objects[start].bbox().centroid()[axis] + objects[end-1].bbox().centroid()[axis]) / 2.0;
                        let mid = objects[start..end].partition_point(|obj| obj.bbox().centroid()[axis] < center);
                        
                        if start >= mid || mid >= end {
                            start + size / 2
                        } else {
                            mid
                        }
                    }
                };

                let left = self.build_bvh(objects, start, mid);
                let right = self.build_bvh(objects, mid, end);
                BVHNode::new_interior_as_box(left, right, bbox, axis as u8)
            }
        }
    }

    fn flatten_bvh(&self, node: &BVHNode) -> Vec<FlatBVHNode> {
        let mut nodes: Vec<FlatBVHNode> = Vec::new();
        self.flatten_bvh_rec(node, &mut nodes);
        nodes
    }

    fn flatten_bvh_rec(&self, node: &BVHNode, nodes: &mut Vec<FlatBVHNode>) {
        match node {
            BVHNode::Interior {
                left,
                right,
                bbox,
                axis,
            } => {
                let offset = nodes.len();
                nodes.push(FlatBVHNode::new_interior(nodes.len() + 1, *bbox, *axis));

                self.flatten_bvh_rec(left, nodes);
                let right_offset = nodes.len();
                self.flatten_bvh_rec(right, nodes);

                match &mut nodes[offset] {
                    FlatBVHNode::Interior {
                        right_child_offset, ..
                    } => {
                        *right_child_offset = right_offset;
                    }
                    _ => panic!("This should not happen"),
                }
            }
            BVHNode::Leaf {
                start,
                size,
                bbox,
            } => {
                nodes.push(FlatBVHNode::new_leaf(*start, *size, *bbox));
            }
        }
    }
}

pub enum BVHNode {
    Interior {
        left: BVHNodePtr,
        right: BVHNodePtr,
        bbox: BBox,
        axis: u8,
    },
    Leaf {
        start: usize,
        size: usize,
        bbox: BBox
    },
}

impl BVHNode {
    #[inline(always)]
    fn new_interior(left: BVHNodePtr, right: BVHNodePtr, bbox: BBox, axis: u8) -> Self {
        BVHNode::Interior {
            left,
            right,
            bbox,
            axis,
        }
    }

    #[inline(always)]
    fn new_leaf(start: usize, size: usize, bbox: BBox) -> Self {
        BVHNode::Leaf {
            start,
            size,
            bbox,
        }
    }

    #[inline(always)]
    fn new_interior_as_box(
        left: BVHNodePtr,
        right: BVHNodePtr,
        bbox: BBox,
        axis: u8,
    ) -> BVHNodePtr {
        Box::new(BVHNode::new_interior(left, right, bbox, axis))
    }

    #[inline(always)]
    fn new_leaf_as_box(start: usize, size: usize, bbox: BBox) -> BVHNodePtr {
        Box::new(BVHNode::new_leaf(start, size, bbox))
    }

    #[inline(always)]
    fn get_bbox(&self) -> BBox {
        match self {
            BVHNode::Interior { bbox, .. } => *bbox,
            BVHNode::Leaf { bbox, .. } => *bbox,
        }
    }

    #[inline(always)]
    fn get_bbox_ref(&self) -> &BBox {
        match self {
            BVHNode::Interior { bbox, .. } => bbox,
            BVHNode::Leaf { bbox, .. } => bbox,
        }
    }

    fn biased_traverse(
        &self,
        ordered_objects: &Vec<Primitive>,
        ray: &Ray,
        ray_t_min: f32,
        ray_t_max: f32,
        other: &Self,
        t_min_other: f32,
    ) -> Option<HitRecord> {
        // Bboxes overlap
        let self_hit = self.traverse(ordered_objects, ray, ray_t_min, ray_t_max);
        match self_hit {
            Some(self_hit) => {
                let self_hit_time = self_hit.t();
                if self_hit_time < t_min_other {
                    Some(self_hit)
                } else {
                    match other.traverse(ordered_objects, ray, ray_t_min, ray_t_max) {
                        Some(other_hit) => {
                            let other_hit_time = other_hit.t();
                            match self_hit_time < other_hit_time {
                                true => Some(self_hit),
                                false => Some(other_hit),
                            }
                        }
                        None => Some(self_hit),
                    }
                }
            }
            None => other.traverse(ordered_objects, ray, ray_t_min, ray_t_max),
        }
    }

    fn traverse(
        &self,
        ordered_objects: &Vec<Primitive>,
        ray: &Ray,
        ray_t_min: f32,
        ray_t_max: f32,
    ) -> Option<HitRecord> {
        match self {
            BVHNode::Interior {
                left,
                right,
                bbox: _,
                axis: _,
            } => {
                let left_intersect = left.get_bbox_ref().intersect(ray, ray_t_min, ray_t_max);
                let right_intersect = right.get_bbox_ref().intersect(ray, ray_t_min, ray_t_max);

                match (left_intersect, right_intersect) {
                    (Some((t_min_l, t_max_l)), Some((t_min_r, t_max_r))) => {
                        match (t_max_l < t_min_r, t_max_r < t_min_l) {
                            (true, true) => {
                                panic!("This should not happen");
                            }
                            (true, false) => left.biased_traverse(
                                ordered_objects,
                                ray,
                                ray_t_min,
                                ray_t_max,
                                right,
                                t_min_r,
                            ),
                            (false, true) => right.biased_traverse(
                                ordered_objects,
                                ray,
                                ray_t_min,
                                ray_t_max,
                                left,
                                t_min_l,
                            ),
                            (false, false) => {
                                // Bboxes overlap
                                if t_min_l < t_min_r {
                                    left.biased_traverse(
                                        ordered_objects,
                                        ray,
                                        ray_t_min,
                                        ray_t_max,
                                        right,
                                        t_min_r,
                                    )
                                } else {
                                    right.biased_traverse(
                                        ordered_objects,
                                        ray,
                                        ray_t_min,
                                        ray_t_max,
                                        left,
                                        t_min_l,
                                    )
                                }
                            }
                        }
                    }
                    (Some((_t_min, _t_max)), None) => {
                        left.traverse(ordered_objects, ray, ray_t_min, ray_t_max)
                    }
                    (None, Some((_t_min, _t_max))) => {
                        right.traverse(ordered_objects, ray, ray_t_min, ray_t_max)
                    }
                    (None, None) => None,
                }
            }
            BVHNode::Leaf {
                start,
                size,
                bbox: _,
            } => {
                let mut closest_hit: Option<HitRecord> = None;
                let mut closest_so_far = ray_t_max;

                for i in *start..(*start + *size) {
                    if let Some(hit) = ordered_objects[i].hit(ray, ray_t_min, closest_so_far) {
                        closest_so_far = hit.t();
                        closest_hit = Some(hit);
                    }
                }

                closest_hit
            }
        }
    }
}

pub enum FlatBVHNode {
    Interior {
        right_child_offset: usize,
        bbox: BBox,
        axis: u8,
    },
    Leaf {
        offset: usize,
        size: usize,
        bbox: BBox,
    },
}

impl FlatBVHNode {
    pub fn new_interior(right_child_offset: usize, bbox: BBox, axis: u8) -> Self {
        FlatBVHNode::Interior {
            right_child_offset,
            bbox,
            axis,
        }
    }

    pub fn new_leaf(offset: usize, size: usize, bbox: BBox) -> Self {
        FlatBVHNode::Leaf {
            offset,
            size,
            bbox
        }
    }

    pub fn get_bbox(&self) -> BBox {
        match self {
            FlatBVHNode::Interior { bbox, .. } => *bbox,
            FlatBVHNode::Leaf { bbox, .. } => *bbox,
        }
    }
}

pub struct BVH {
    root: BVHNodePtr,
    ordered_objects: Vec<Primitive>,
}

impl Hittable for BVH {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if self
            .root
            .get_bbox_ref()
            .intersect(ray, t_min, t_max)
            .is_none()
        {
            return None;
        }
        self.root.traverse(&self.ordered_objects, ray, t_min, t_max)
    }
    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        None
    }
    fn bbox(&self) -> BBox {
        self.root.get_bbox()
    }
}

pub struct FlatBVH {
    nodes: Box<[FlatBVHNode]>,
    ordered_objects: Vec<Primitive>,
}

impl FlatBVH {
    pub fn traverse(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        let mut current_node_index = 0;
        let mut to_visit_offset = 0;
        let mut nodes_to_visit: [usize; 64] = [0; 64];

        let dir_is_neg = [
            ray.direction.x < 0.0,
            ray.direction.y < 0.0,
            ray.direction.z < 0.0,
        ];

        // TODO: refactor into bbox aware traversal
        loop {
            let node = &self.nodes[current_node_index];
            match node.get_bbox().intersect(ray, t_min, t_max) {
                Some((_, _)) => {
                    match node {
                        FlatBVHNode::Interior {
                            right_child_offset,
                            bbox: _,
                            axis,
                        } => {
                            if dir_is_neg[*axis as usize] {
                                nodes_to_visit[to_visit_offset] = current_node_index + 1;
                                to_visit_offset += 1;
                                current_node_index = *right_child_offset;
                            } else {
                                nodes_to_visit[to_visit_offset] = *right_child_offset;
                                to_visit_offset += 1;
                                current_node_index += 1;
                            }
                        }
                        FlatBVHNode::Leaf {
                            offset,
                            size,
                            bbox: _,
                        } => {
                                for i in *offset..(*offset + *size) {
                                    if let Some(hit) =
                                        self.ordered_objects[i].hit(ray, t_min, closest_so_far)
                                    {
                                        closest_so_far = hit.t();
                                        closest_hit = Some(hit);
                                    }
                                }
                                if to_visit_offset == 0 {
                                    break;
                                }
                                to_visit_offset -= 1;
                                current_node_index = nodes_to_visit[to_visit_offset];
                            }

                    }
                }
                None => {
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                }
            }
        }

        closest_hit
    }
}

impl Hittable for FlatBVH {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.traverse(ray, t_min, t_max)
    }
    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        None
    }
    fn bbox(&self) -> BBox {
        match &self.nodes[0] {
            FlatBVHNode::Interior { bbox, .. } => *bbox,
            FlatBVHNode::Leaf { bbox, .. } => *bbox,
        }
    }
}
