use super::Material;
use crate::objects::HitRecord;
use crate::textures::Solid;
use crate::textures::TextureRef;
use crate::types::color::ColorOps;
use crate::types::pdf::{PDF, CosineWeightedHemispherePDF, SpherePDF};
use crate::types::{color::Color, ray::Ray};
use crate::types::onb::OrthonormalBasis;
use rand::rngs::ThreadRng;
use std::sync::Arc;

pub struct Lambertian {
    texture: TextureRef,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Self {
            texture: Arc::new(Solid::new(albedo)),
        }
    }

    pub fn new_texture(texture: Option<TextureRef>) -> Self {
        Self {
            texture: texture.unwrap_or(Arc::new(Solid::new(Color::gray(0.5)))),
        }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        _ray_in: &Ray,
        rec: &HitRecord,
    ) -> Option<(Color, Ray)> {
        let cosine_pdf = CosineWeightedHemispherePDF::new(&rec.normal());

        // Scatter
        let scatter_direction = &cosine_pdf.generate(rng);
        let scattered = Ray::new(rec.p(), scatter_direction.normalize());

        // Attenuation
        let attenuation = self.texture.value(rec.u(), rec.v(), &rec.p());

        Some((attenuation, scattered))
    }

    fn scattering_pdf(&self, _ray_in: &Ray, ray_out: &Ray, rec: &HitRecord) -> f32 {
        let cosine_pdf = CosineWeightedHemispherePDF::new(&rec.normal());
        cosine_pdf.value(&ray_out.direction)
    }
}
