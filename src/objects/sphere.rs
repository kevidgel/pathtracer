use std::sync::Arc;
use crate::objects::Hittable;
use crate::materials::Material;
use crate::types::ray::Ray;
use na::{Point3, Vector3};

use super::HitRecord;

pub struct Sphere {
    center: Point3<f32>,
    radius: f32,
    mat: Option<Arc<dyn Material + Send + Sync>>,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32, mat: Option<Arc<dyn Material + Send + Sync>>) -> Self {
        Self { center, radius , mat }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Solve the quadratic equation
        let oc: Vector3<f32> = self.center - ray.origin;
        let a: f32 = ray.direction.norm_squared();
        let h = ray.direction.dot(&oc);
        let c = oc.norm_squared() - self.radius * self.radius;

        // Discriminant
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            // Miss
            return None;
        }

        let sqrt_disc = discriminant.sqrt();

        // Check whether there is a root in the bounds
        let mut root = (h - sqrt_disc) / a;
        if root < t_min || t_max < root {
            root = (h + sqrt_disc) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let rec = HitRecord::new(
            ray,
            ray.at(root),
            (ray.at(root) - self.center) / self.radius,
            root,
            self.mat.clone(),
        );

        Some(rec)
    }

    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        self.mat.clone()
    }
}
