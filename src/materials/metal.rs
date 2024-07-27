use rand::rngs::ThreadRng;
use super::Material;
use super::reflect;
use crate::types::{ray::Ray, color::Color, color::ColorOps, sampler::{SphereSampler, Sampler}};
use crate::objects::HitRecord;
use na::Vector3;

pub struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f32) -> Self {
        Self { albedo , fuzz: fuzz.clamp(0.0, 1.0) }
    }
}

impl Material for Metal {
    fn scatter(&self, rng: Option<&mut ThreadRng>, ray_in: &Ray, rec: &HitRecord) -> (Color, Ray) {
        let fuzz: Vector3<f32> = if self.fuzz > 0.0 {
            match rng {
                Some(rng) => {
                    let sampler = SphereSampler::unit();
                    self.fuzz * sampler.sample(rng).normalize()
                }
                None => Vector3::zeros()
            }
        } else {
            Vector3::zeros()
        };
        
        let reflected = reflect(&ray_in.direction.normalize(), &rec.normal()) + fuzz;
        let scattered = Ray::new(rec.p(), reflected);
        let attenuation = self.albedo;
        (attenuation, scattered)
    }
}