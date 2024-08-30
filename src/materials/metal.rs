use super::reflect_y;
use crate::types::color::Color;
use super::Material;
use crate::objects::HitRecord;
use crate::types::pdf::{SpherePDF, PDF};
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
        true
    }
    
    fn scatter(&self, rng: &mut ThreadRng, w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        let fuzz: Vector3<f32> = if self.fuzz > 0.0 {
            let sampler = SpherePDF::new();
            self.fuzz * sampler.generate(rng).normalize()
        } else {
            Vector3::zeros()
        };

        let reflected = reflect_y(&w_out.normalize()) + fuzz;

        reflected
    }

    fn bsdf_evaluate(&self, _w_out: &Vector3<f32>, _w_in: &Vector3<f32>, _rec: &HitRecord) -> Color {
        self.albedo
    }
}
