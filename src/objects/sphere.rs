use crate::bvh::BBox;
use crate::materials::MaterialRef;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Point3, Vector3};

use super::HitRecord;

#[repr(align(32))]
#[derive(Clone)]
pub struct Sphere {
    center: Point3<f32>,
    radius: f32,
    mat: Option<MaterialRef>,
    bbox: BBox,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32, mat: Option<MaterialRef>) -> Self {
        let rvec = Vector3::new(radius, radius, radius);
        Self {
            center,
            radius,
            mat,
            bbox: BBox::new(center - rvec, center + rvec),
        }
    }

    fn get_uv(&self, p: &Vector3<f32>) -> (f32, f32) {
        let theta = (-p.y).acos();
        let phi = f32::atan2(-p.z, p.x) + std::f32::consts::PI;
        (
            phi / (2.0 * std::f32::consts::PI),
            theta / std::f32::consts::PI,
        )
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &mut Ray) -> Option<HitRecord> {
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
        if root < ray.t_min || ray.t_max < root {
            root = (h + sqrt_disc) / a;
            if root < ray.t_min || ray.t_max < root {
                return None;
            }
        }

        // Get the normal
        let outward_normal = (ray.at(root) - self.center) / self.radius;
        let (u, v) = self.get_uv(&outward_normal);

        // Hit
        ray.t_max = root;
        let rec = HitRecord::new(
            ray,
            ray.at(root),
            outward_normal,
            root,
            self.mat.clone(),
            u,
            v,
        );

        Some(rec)
    }

    fn mat(&self) -> Option<MaterialRef> {
        self.mat.clone()
    }

    fn bbox(&self) -> BBox {
        self.bbox
    }
}
