use crate::materials::Material;
use crate::types::{ray::Ray, color::Color};
use crate::objects::HitRecord;
use crate::types::sampler::{SphereSampler, Sampler};
use rand::Rng;
use na::Vector3;

pub struct Lambertian {
    albedo: Color,
}

impl Material for Lambertian {
    fn scatter(&self, rng: Option<&mut impl Rng>, _ray_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let sampler = SphereSampler::unit();

        let mut scatter_direction = rec.normal() + sampler.sample(rng);

        let scattered = Ray::new(rec.p, scatter_direction);
        let attenuation = self.albedo;

        Some((attenuation, scattered))
    }
}
