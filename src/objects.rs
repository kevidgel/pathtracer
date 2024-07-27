pub mod sphere;

use std::sync::Arc;
use crate::types::ray::Ray;
use crate::materials::Material;
use na::{Point3, Vector3};

pub struct HitRecord {
    p: Point3<f32>,
    normal: Vector3<f32>,
    t: f32,
    front_face: bool,
    material: Option<Arc<dyn Material + Sync + Send>>,
}

impl HitRecord {
    pub fn new(ray: &Ray, p: Point3<f32>, normal: Vector3<f32>, t: f32, material: Option<Arc<dyn Material + Sync + Send>>) -> HitRecord {
        let mut rec: HitRecord = HitRecord {
            p,
            normal,
            t,
            front_face: false,
            material,
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
}

// All of our things that are hittable
pub struct HittableObjects {
    // TODO: replace with kd tree
    // TODO: idk if we should box this 
    objs: Vec<Arc<dyn Hittable + Sync + Send>>,
}

impl HittableObjects {
    pub fn new() -> HittableObjects {
        HittableObjects { objs: Vec::new() }
    }

    pub fn add(&mut self, obj: Arc<dyn Hittable + Sync + Send>) {
        self.objs.push(obj);
    }

    pub fn clear(&mut self) {
        self.objs.clear();
    }
}

impl Hittable for HittableObjects {
    // Ideally we replace this with spatial data structure
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for obj in &self.objs {
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
}