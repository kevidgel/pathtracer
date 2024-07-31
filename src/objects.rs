pub mod sphere;
pub mod tri_mesh;

use crate::bvh::BBox;
use crate::materials::Material;
use crate::types::ray::Ray;
use na::{Point3, Vector3};
use std::sync::Arc;

pub type Primitive = Arc<dyn Hittable + Sync + Send>;

pub struct HitRecord {
    // Normal stuff
    p: Point3<f32>,
    normal: Vector3<f32>,
    t: f32,
    front_face: bool,

    // Material
    material: Option<Arc<dyn Material + Sync + Send>>,

    // Texture
    u: f32,
    v: f32,
}

impl HitRecord {
    pub fn new(
        ray: &Ray,
        p: Point3<f32>,
        normal: Vector3<f32>,
        t: f32,
        material: Option<Arc<dyn Material + Sync + Send>>,
        u: f32,
        v: f32,
    ) -> HitRecord {
        let mut rec: HitRecord = HitRecord {
            p,
            normal,
            t,
            front_face: false,
            material,
            u,
            v,
        };

        rec.set_face_normal(ray, normal);

        rec
    }

    pub fn p(&self) -> Point3<f32> {
        self.p
    }

    pub fn normal(&self) -> Vector3<f32> {
        self.normal
    }

    pub fn t(&self) -> f32 {
        self.t
    }

    pub fn front_face(&self) -> bool {
        self.front_face
    }

    pub fn material(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        self.material.clone()
    }

    pub fn u(&self) -> f32 {
        self.u
    }

    pub fn v(&self) -> f32 {
        self.v
    }

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector3<f32>) {
        // Flip the normal if the ray is inside the object
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

// We love traits !!!
pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>>;
    fn bbox(&self) -> BBox;
}

// All of our things that are hittable
#[derive(Clone)]
pub struct HittableObjects {
    // TODO: replace with kd tree
    // TODO: idk if we should box this
    objs: Vec<Primitive>,
    bbox: BBox,
}

impl HittableObjects {
    pub fn new() -> HittableObjects {
        HittableObjects {
            objs: Vec::new(),
            bbox: BBox::empty(),
        }
    }

    pub fn objs_clone(&self) -> Vec<Primitive> {
        self.objs.clone()
    }

    pub fn add(&mut self, obj: Primitive) {
        self.bbox = self.bbox.merge(&obj.bbox());
        self.objs.push(obj);
    }

    pub fn clear(&mut self) {
        self.objs.clear();
    }
}

impl Hittable for HittableObjects {
    // Ideally we replace this with spatial data structure
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if self.bbox.intersect(ray, t_min, t_max).is_none() {
            return None;
        }

        let mut rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for obj in &self.objs {
            if obj.bbox().intersect(ray, t_min, closest_so_far).is_none() {
                continue;
            }
            if let Some(hit) = obj.hit(ray, t_min, closest_so_far) {
                closest_so_far = hit.t();
                rec = Some(hit);
            }
        }
        rec
    }

    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        None
    }

    fn bbox(&self) -> BBox {
        self.bbox
    }
}
