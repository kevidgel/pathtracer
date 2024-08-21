use super::Material;
use super::{reflect, refract};
use crate::objects::HitRecord;
use crate::types::{color::Color, color::ColorOps, ray::Ray};
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
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        ray_in: &Ray,
        rec: &HitRecord,
    ) -> Option<(Color, Ray)> {
        let attenuation = Color::gray(1.0_f32);
        let ri: f32 = if rec.front_face() {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };

        let unit_direction = ray_in.direction.normalize();
        let cos_theta = (-unit_direction).dot(&rec.normal()).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let reflect_thresh: f32 = rng.gen_range(0.0..1.0);

        let cannot_refract = ri * sin_theta > 1.0;
        let direction =
            match cannot_refract || Dielectric::reflectance(cos_theta, ri) > reflect_thresh {
                true => reflect(&unit_direction, &rec.normal()),
                false => refract(&unit_direction, &rec.normal(), ri),
            };

        let scattered = Ray::new(rec.p(), direction);
        Some((attenuation, scattered))
    }
}
