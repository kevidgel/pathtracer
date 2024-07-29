use crate::materials::Material;
use crate::objects::{HitRecord, Hittable, HittableObjects};
use crate::types::ray::Ray;
use na::{Point3, Vector3};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::sync::Arc;

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

pub enum AxisMethod {
    Random,
    LongestAxisFirst,
}

pub struct BVHNode {
    left: Arc<dyn Hittable + Sync + Send>,
    right: Arc<dyn Hittable + Sync + Send>,
    bbox: BBox,
}

impl BVHNode {
    pub fn build_from_hittable_objects(
        rng: &mut Option<&mut ThreadRng>,
        method: AxisMethod,
        objects: HittableObjects,
    ) -> Self {
        // We clone objects to avoid mutating the original
        let mut objects = objects.objs_clone();
        let len = objects.len();

        Self::build(rng, &method, &mut objects, 0, len)
    }

    fn get_axis(
        rng: &mut Option<&mut ThreadRng>,
        method: &AxisMethod,
        objects: &mut Vec<Arc<dyn Hittable + Sync + Send>>,
        start: usize,
        end: usize,
    ) -> usize {
        match method {
            AxisMethod::Random => match rng {
                Some(rng) => rng.gen_range(0..3),
                None => {
                    let mut rng = thread_rng();

                    rng.gen_range(0..3)
                }
            },
            AxisMethod::LongestAxisFirst => {
                // not very functional hahah
                let mut bbox = BBox::empty();
                for i in start..end {
                    bbox = bbox.merge(&objects[i].bbox());
                }

                bbox.get_longest_axis()
            }
        }
    }

    fn bbox_comp(
        a: &Arc<dyn Hittable + Sync + Send>,
        b: &Arc<dyn Hittable + Sync + Send>,
        axis: usize,
    ) -> std::cmp::Ordering {
        let a_bbox = a.bbox();
        let b_bbox = b.bbox();

        if a_bbox.min[axis] < b_bbox.min[axis] {
            std::cmp::Ordering::Less
        } else if a_bbox.min[axis] > b_bbox.min[axis] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    pub fn build(
        rng: &mut Option<&mut ThreadRng>,
        method: &AxisMethod,
        objects: &mut Vec<Arc<dyn Hittable + Sync + Send>>,
        start: usize,
        end: usize,
    ) -> Self {
        let axis = Self::get_axis(rng, &method, objects, start, end);

        let range = end - start;
        // TODO: Set max leaf size.
        let (left, right) = match range {
            1 => {
                let left = objects[start].clone();
                let right = objects[start].clone();

                (left, right)
            }
            2 => {
                let left = objects[start].clone();
                let right = objects[start + 1].clone();

                (left, right)
            }
            _ => {
                objects[start..end].sort_by(|a, b| Self::bbox_comp(a, b, axis));
                // TODO: SAH heurstic
                let mid = Self::find_partition(objects, start, end);
                let mid = start + range / 2;

                let left = Arc::new(Self::build(rng, method, objects, start, mid))
                    as Arc<dyn Hittable + Sync + Send>;
                let right = Arc::new(Self::build(rng, method, objects, mid, end))
                    as Arc<dyn Hittable + Sync + Send>;

                (left, right)
            }
        };

        let bbox = left.bbox().merge(&right.bbox());

        Self { left, right, bbox }
    }

    fn find_partition(
        objects: &mut Vec<Arc<dyn Hittable + Sync + Send>>,
        start: usize,
        end: usize,
    ) -> usize {
        let mut cost = std::f32::MAX;
        let mut mid = start + (end - start) / 2;
        let bins = 20;
        let step = (end - start) / bins;
        for i in (start..end).step_by(step.max(1)) {
            let new_cost = Self::cost(objects, start, i, end);
            if new_cost < cost {
                cost = new_cost;
                mid = i;
            }
        }

        mid
    }

    fn cost(objects: &mut Vec<Arc<dyn Hittable + Sync + Send>>, start: usize, mid: usize, end: usize) -> f32 { 
        let mut left = BBox::empty();
        let mut right = BBox::empty();

        for i in start..mid {
            left = left.merge(&objects[i].bbox());
        }
        for i in mid..end {
            right = right.merge(&objects[i].bbox());
        }

        return left.get_surface_area() * (mid - start) as f32 + right.get_surface_area() * (end - mid) as f32;
    }
}

impl Hittable for BVHNode {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if !self.bbox.hit(ray, t_min, t_max) {
            return None;
        }

        let hit_left = self.left.hit(ray, t_min, t_max);
        let hit_right = self.right.hit(ray, t_min, t_max);

        match (hit_left, hit_right) {
            (Some(hit_left), Some(hit_right)) => {
                if hit_left.t() < hit_right.t() {
                    Some(hit_left)
                } else {
                    Some(hit_right)
                }
            }
            (Some(hit_left), None) => Some(hit_left),
            (None, Some(hit_right)) => Some(hit_right),
            (None, None) => None,
        }
    }
    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        None
    }
    fn bbox(&self) -> BBox {
        self.bbox
    }
}
