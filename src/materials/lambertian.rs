use super::Material;
use na::Vector3;
use crate::objects::HitRecord;
use crate::textures::Solid;
use crate::textures::TextureRef;
use crate::types::color::ColorOps;
use crate::types::onb::OrthonormalBasis;
use crate::types::pdf::{CosineWeightedHemispherePDF, SpherePDF, PDF};
use crate::types::{color::Color, ray::Ray};
use rand::rngs::ThreadRng;
use rand::Rng;
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
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn scatter(&self, rng: &mut ThreadRng, _w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        Vector3::new(x, y, z)
    }

    fn bsdf_evaluate(&self, _w_out: &Vector3<f32>, _w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        self.texture.value(rec.u(), rec.v(), &rec.p()) / std::f32::consts::PI 
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        &w_in.normalize().y / std::f32::consts::PI    
    }
}
