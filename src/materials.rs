pub mod dielectric;
pub mod lambertian;
pub mod metal;

use crate::objects::HitRecord;
use crate::types::{color::Color, ray::Ray};
use na::Vector3;
use rand::rngs::ThreadRng;
use std::collections::BTreeMap;
use std::sync::Arc;

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
    fn scatter(&self, rng: Option<&mut ThreadRng>, ray_in: &Ray, rec: &HitRecord) -> (Color, Ray);
}

pub struct MaterialRegistry {
    materials: BTreeMap<String, Arc<dyn Material + Send + Sync>>,
}

impl MaterialRegistry {
    pub fn new() -> Self {
        MaterialRegistry {
            materials: BTreeMap::new(),
        }
    }

    pub fn add_material(&mut self, name: &str, material: Arc<dyn Material + Send + Sync>) {
        self.materials.insert(name.to_string(), material);
    }

    pub fn create_material(
        &mut self,
        name: &str,
        material: impl Material + std::marker::Send + std::marker::Sync + 'static,
    ) {
        let material: Arc<dyn Material + Sync + Send> = Arc::new(material);
        self.add_material(name, material);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Material + Send + Sync>> {
        match self.materials.get(name) {
            Some(material) => Some(material.clone()),
            None => {
                log::error!("Material not found: {}", name);
                None
            }
        }
    }
}
