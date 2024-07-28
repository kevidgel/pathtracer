use super::Material;
use crate::objects::HitRecord;
use crate::textures::Solid;
use crate::textures::Texture;
use crate::types::color::ColorOps;
use crate::types::sampler::{Sampler, SphereSampler};
use crate::types::{color::Color, ray::Ray};
use rand::rngs::ThreadRng;
use std::sync::Arc;

pub struct Lambertian {
    texture: Arc<dyn Texture + Send + Sync>,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Self {
            texture: Arc::new(Solid::new(albedo)),
        }
    }

    pub fn new_texture(texture: Option<Arc<dyn Texture + Send + Sync>>) -> Self {
        Self {
            texture: texture.unwrap_or(Arc::new(Solid::new(Color::gray(0.5)))),
        }
    }
}

impl Material for Lambertian {
    fn scatter(&self, rng: Option<&mut ThreadRng>, _ray_in: &Ray, rec: &HitRecord) -> (Color, Ray) {
        let sampler = SphereSampler::unit();

        let rng = match rng {
            Some(rng) => rng,
            None => &mut rand::thread_rng(),
        };

        // Scatter
        let scatter_direction = rec.normal() + sampler.sample(rng).normalize();

        // Reject small offsets
        let scattered: Ray = match scatter_direction.x < f32::EPSILON
            && scatter_direction.y < f32::EPSILON
            && scatter_direction.z < f32::EPSILON
        {
            true => Ray::new(rec.p(), rec.normal()),
            false => Ray::new(rec.p(), scatter_direction),
        };

        // Attenuation

        let attenuation = self.texture.value(rec.u(), rec.v(), &rec.p());

        (attenuation, scattered)
    }
}

pub struct Diffuse {
    albedo: Color,
}

impl Diffuse {
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Material for Diffuse {
    fn scatter(&self, rng: Option<&mut ThreadRng>, _ray_in: &Ray, rec: &HitRecord) -> (Color, Ray) {
        let sampler = SphereSampler::unit();

        let rng = match rng {
            Some(rng) => rng,
            None => &mut rand::thread_rng(),
        };

        // Scatter
        let scatter_direction = sampler.sample_on_hemisphere(rng, &rec.normal());

        let scattered: Ray = match scatter_direction.x < f32::EPSILON
            && scatter_direction.y < f32::EPSILON
            && scatter_direction.z < f32::EPSILON
        {
            true => Ray::new(rec.p(), rec.normal()),
            false => Ray::new(rec.p(), scatter_direction),
        };

        // Attenuation
        let attenuation = self.albedo;

        (attenuation, scattered)
    }
}
