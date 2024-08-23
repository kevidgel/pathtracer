use core::f32;

use super::reflect;
use super::Material;
use crate::objects::HitRecord;
use crate::types::{
    color::Color,
    ray::Ray,
    sampler::{Sampler, SphereSampler},
};
use na::Vector3;
use rand::rngs::ThreadRng;

pub struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f32) -> Self {
        Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn is_emissive(&self) -> bool {
        false
    }
    fn is_specular(&self) -> bool {
        false
    }
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        ray_in: &Ray,
        rec: &HitRecord,
    ) -> Option<Ray> {
        let fuzz: Vector3<f32> = if self.fuzz > 0.0 {
            let sampler = SphereSampler::unit();
            self.fuzz * sampler.sample(rng).normalize()
        } else {
            Vector3::zeros()
        };

        let reflected = reflect(&ray_in.direction.normalize(), &rec.normal()) + fuzz;
        let scattered = Ray::new(rec.p(), reflected);
        
        Some(scattered)
    }

    fn bsdf_evaluate(&self, _ray_in: &Ray, _ray_out: &Ray, _rec: &HitRecord) -> Color {
        self.albedo
    }
}
