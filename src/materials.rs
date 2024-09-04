pub mod dielectric;
pub mod disney;
pub mod lambertian;
pub mod light;
pub mod metal;

use crate::objects::HitRecord;
use crate::types::{color::Color, ray::Ray};
use na::{Point3, Vector3};
use rand::rngs::ThreadRng;
use std::collections::BTreeMap;
use std::sync::Arc;

pub type MaterialRef = Arc<dyn Material>;

pub fn reflect(v: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
    let v = -v;
    v - 2.0 * v.dot(n) * n
}

pub fn reflect_y(v: &Vector3<f32>) -> Vector3<f32> {
    let mut out = v.clone();
    out.y = -out.y;

    out
}

pub fn refract(w_o: &Vector3<f32>, n: &Vector3<f32>, eta: f32) -> Option<Vector3<f32>> {
    let cos_o = w_o.dot(&n).min(1.0);
    let sin2_o = 1.0 - cos_o * cos_o;
    let sin2_i = sin2_o * eta * eta;

    if sin2_i >= 1.0 {
        None
    } else {
        let cos_i = (1.0 - sin2_i).sqrt();

        Some(((eta * -w_o) + ((eta * cos_o) - cos_i) * n).normalize())
    }
}

#[cfg(test)]
mod tests {
    use crate::types::color::ColorOps;

    use super::*;

    #[test]
    fn test_refract() {
        let test_in = Vector3::new(0.7071068, 0.7071068, 0.0);
        let test_normal = Vector3::new(0.0, 1.0, 0.0);

        println!("{}", refract(&test_in, &test_normal, 1.0 / 1.5).unwrap());
    }

    #[test]
    fn test_refract2() {
        let eta = 1.0 / 1.5;
        let w_o = Vector3::random().normalize();
        let h = Vector3::random().normalize();
        let w_i = refract(&w_o, &h, eta).unwrap().normalize();

        println!("{}", h);
        println!("{}", (w_o + (1.0 / eta) * w_i).normalize());
    }
}

pub fn refract_y(uv: &Vector3<f32>, etai_over_etat: f32) -> Vector3<f32> {
    let cos_theta = -uv.y.min(1.0);
    let r_out_perp = etai_over_etat * (uv + (cos_theta * Vector3::new(0.0, 1.0, 0.0)));
    let r_out_parallel =
        -(1.0 - r_out_perp.norm_squared()).abs().sqrt() * Vector3::new(0.0, 1.0, 0.0);
    r_out_perp + r_out_parallel
}

pub struct ScatterRecord {
    pub w_in: Vector3<f32>,
}

pub trait Material: Send + Sync {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn scatter(
        &self,
        _rng: &mut ThreadRng,
        _w_out: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> ScatterRecord {
        ScatterRecord {
            w_in: Vector3::zeros(),
        }
    }

    fn bsdf_evaluate(
        &self,
        _w_out: &Vector3<f32>,
        _w_in: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> Color {
        Color::zeros()
    }

    fn emitted(
        &self,
        _ray_out: &Ray,
        _rec: &HitRecord,
        _u: f32,
        _v: f32,
        _p: &Point3<f32>,
    ) -> Color {
        Color::zeros()
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, _w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        0.0
    }
}

pub struct MaterialRegistry {
    materials: BTreeMap<String, MaterialRef>,
}

impl MaterialRegistry {
    pub fn new() -> Self {
        MaterialRegistry {
            materials: BTreeMap::new(),
        }
    }

    pub fn add_material(&mut self, name: &str, material: MaterialRef) {
        self.materials.insert(name.to_string(), material);
    }

    pub fn create_material(
        &mut self,
        name: &str,
        material: impl Material + std::marker::Send + std::marker::Sync + 'static,
    ) {
        let material: MaterialRef = Arc::new(material);
        self.add_material(name, material);
    }

    pub fn get(&self, name: &str) -> Option<MaterialRef> {
        match self.materials.get(name) {
            Some(material) => Some(material.clone()),
            None => {
                log::error!("Material not found: {}", name);
                None
            }
        }
    }
}
