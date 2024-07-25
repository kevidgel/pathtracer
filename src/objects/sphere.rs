extern crate nalgebra as na;

use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Point3, Vector3};

use super::HitRecord;

#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub center: Point3<f32>,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, rec: &mut HitRecord) -> bool {
        // Solve the quadratic equation
        let oc: Vector3<f32> = self.center - ray.origin;
        let a: f32 = ray.direction.norm_squared();
        let h = ray.direction.dot(&oc);
        let c = oc.norm_squared() - self.radius * self.radius;

        // Discriminant
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            // Miss
            return false;
        }

        let sqrt_disc = discriminant.sqrt();

        // Check whether there is a root in the bounds
        let mut root = (h - sqrt_disc) / a;
        if root < t_min || t_max < root {
            root = (h + sqrt_disc) / a;
            if root < t_min || t_max < root {
                return false;
            }
        }

        rec.t = root;
        rec.p = ray.at(rec.t);
        rec.normal = (rec.p - self.center) / self.radius;

        return true;
    }
}
