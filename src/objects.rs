extern crate nalgebra as na;
pub mod sphere;

use crate::types::ray::Ray;
use na::Point3;

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub p: Point3<f32>,
    pub normal: na::Vector3<f32>,
    pub t: f32,
}

impl HitRecord {
    pub fn new() -> HitRecord {
        HitRecord {
            p: Point3::new(0.0, 0.0, 0.0),
            normal: na::Vector3::new(0.0, 0.0, 0.0),
            t: 0.0,
        }
    }
}

// We love traits !!!
pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord) -> bool;
}
