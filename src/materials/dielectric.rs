use super::*;
use crate::materials::disney::fresnel_o;
use crate::objects::HitRecord;
use crate::types::{color::Color, color::ColorOps, ray::Ray};
use na::Vector3;
use rand::rngs::ThreadRng;
use rand::Rng;

pub struct Dielectric {
    ref_idx: f32,
}

impl Dielectric {
    pub fn new(ref_idx: f32) -> Self {
        Self { ref_idx }
    }

    pub fn reflectance(cos: f32, ref_idx: f32) -> f32 {
        let r0 = (1.0_f32 - ref_idx) / (1.0_f32 + ref_idx);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cos).powf(5.0)
    }
}

impl Material for Dielectric {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        true
    }

    fn scatter(&self, rng: &mut ThreadRng, w_out: &Vector3<f32>, rec: &HitRecord) -> ScatterRecord {
        let ri: f32 = if rec.front_face() {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };

        let unit_direction = -w_out.normalize();
        let cos_theta = -unit_direction.y.min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let reflect_thresh: f32 = rng.gen_range(0.0..1.0);

        let cannot_refract = ri * sin_theta > 1.0;
        let fresnel = Dielectric::reflectance(cos_theta, ri);
        let fresnel = fresnel_o(&unit_direction, &Vector3::new(0.0, 1.0, 0.0), ri);

        let direction = match cannot_refract || fresnel > reflect_thresh {
            true => reflect_y(&unit_direction),
            false => refract_y(&unit_direction, ri),
        };

        ScatterRecord { w_in: direction }
    }

    fn bsdf_evaluate(
        &self,
        _w_out: &Vector3<f32>,
        _w_in: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> Color {
        Color::gray(1.0)
    }
}
