use crate::materials::MaterialRef;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use crate::{bvh::BBox, types::onb::OrthonormalBasis};
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

    fn random_to_sphere(
        &self,
        rng: &mut impl rand::Rng,
        radius: f32,
        distance_squared: f32,
    ) -> Vector3<f32> {
        let r1 = rng.gen_range(0.0..1.0);
        let r2 = rng.gen_range(0.0..1.0);
        let z = 1.0 + r2 * ((1.0 - self.radius * self.radius / distance_squared).sqrt() - 1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * (1.0 - z * z).sqrt();
        let y = phi.sin() * (1.0 - z * z).sqrt();

        Vector3::new(x, y, z) * radius
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

    fn pdf(&self, ray: &Ray) -> f32 {
        match self.hit(&mut ray.clone()) {
            Some(_rec) => {
                let cos_theta_max = (1.0
                    - self.radius * self.radius / (self.center - ray.origin).norm_squared())
                .sqrt();
                let solid_angle = 2.0 * std::f32::consts::PI * (1.0 - cos_theta_max);
                1.0 / solid_angle
            }
            None => 0.0,
        }
    }

    fn sample(&self, rng: &mut impl rand::Rng, origin: &Point3<f32>) -> Vector3<f32> {
        let direction = self.center - origin;
        let distance_squared = direction.norm_squared();
        let uvw = OrthonormalBasis::new(&direction);

        uvw.to_world(&self.random_to_sphere(rng, self.radius, distance_squared))
    }
}
