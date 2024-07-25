extern crate nalgebra as na;
pub mod sphere;

use crate::types::ray::Ray;
use na::{Point3, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub p: Point3<f32>,
    pub normal: Vector3<f32>,
    pub t: f32,
    pub front_face: bool,
}

impl HitRecord {
    pub fn new(ray: &Ray, p: Point3<f32>, normal: Vector3<f32>, t: f32) -> HitRecord {
        let mut rec: HitRecord = HitRecord {
            p,
            normal,
            t,
            front_face: false,
        };

        rec.set_face_normal(ray, normal);

        return rec;
    }

    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: na::Vector3<f32>) {
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
}
