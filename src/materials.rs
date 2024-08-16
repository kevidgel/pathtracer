pub mod dielectric;
pub mod lambertian;
pub mod light;
pub mod metal;

use crate::objects::HitRecord;
use crate::types::color::ColorOps;
use crate::types::{color::Color, ray::Ray};
use na::{Point3, Vector3};
use rand::rngs::ThreadRng;
use std::collections::BTreeMap;
use std::sync::Arc;

pub type MaterialRef = Arc<dyn Material + Send + Sync>;

pub fn reflect(v: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(n) * n
}

pub fn refract(uv: &Vector3<f32>, n: &Vector3<f32>, etai_over_etat: f32) -> Vector3<f32> {
    let cos_theta = -uv.dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + (cos_theta * n));
    let r_out_parallel = -(1.0 - r_out_perp.norm_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}

pub trait Material {
    fn scatter(
        &self,
        rng: Option<&mut ThreadRng>,
        ray_in: &Ray,
        rec: &HitRecord,
    ) -> Option<(Color, Ray)>;
    fn emitted(&self, _u: f32, _v: f32, _p: &Point3<f32>) -> Color {
        Color::zeros()
    }
    fn pdf(&self, _ray_in: &Ray, _ray_out: &Ray, _rec: &HitRecord) -> f32 {
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
