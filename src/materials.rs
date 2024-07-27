pub mod lambertian;
pub mod metal;
pub mod dielectric;

use std::collections::BTreeMap;
use std::sync::Arc;
use crate::types::{ray::Ray, color::Color};
use crate::objects::HitRecord;
use crate::config::FromConfig;
use rand::rngs::ThreadRng;
use na::Vector3;
use config::{Config, Value};

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
        MaterialRegistry { materials: BTreeMap::new() }
    }

    pub fn add_material(&mut self, name: &str, material: Arc<dyn Material + Send + Sync>) {
        self.materials.insert(name.to_string(), material);
    }

    pub fn create_material(&mut self, name: &str, material: impl Material + std::marker::Send + std::marker::Sync + 'static) {
        let material: Arc<dyn Material + Sync + Send> = Arc::new(material);
        self.add_material(name, material);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Material + Send + Sync>> {
        self.materials.get(name).map(|m| m.clone())
    }
}

// impl FromConfig for MaterialRegistry {
//     fn build(config: Config) -> Self {
//         let mut registry = MaterialRegistry::new();
//         let materials = config.get("materials").expect("No materials found in config");
//         for (name, material) in materials.as_mapping().expect("Materials must be a mapping") {
//             let material = material.as_mapping().expect("Material must be a mapping");
//             let material_type = material.get("type").expect("Material must have a type").as_str().expect("Material type must be a string");
//             match material_type {
//                 "lambertian" => {
//                     let albedo = material.get("albedo").expect("Lambertian material must have an albedo").as_sequence().expect("Albedo must be a sequence");
//                     let albedo = Color::from_config(&serde_yaml::Value::Sequence(albedo.clone()));
//                     let material = lambertian::Lambertian::new(albedo);
//                     registry.create_material(name.as_str().expect("Material name must be a string"), material);
//                 }
//                 "metal" => {
//                     let albedo = material.get("albedo").expect("Metal material must have an albedo").as_sequence().expect("Albedo must be a sequence");
//                     let albedo = Color::from_config(&serde_yaml::Value::Sequence(albedo.clone()));
//                     let fuzz = material.get("fuzz").expect("Metal material must have a fuzz value").as_f64().expect("Fuzz value must be a float") as f32;
//                     let material = metal::Metal::new(albedo, fuzz);
//                     registry.create_material(name.as_str().expect("Material name must be a string"), material);
//                 }
//                 "dielectric" => {
//                     let ref_idx = material.get("ref_idx").expect("Dielectric material must have a ref_idx").as_f64().expect("Ref_idx must be a float") as f32;
//                     let material = dielectric::Dielectric::new(ref_idx);
//                     registry.create_material(name.as_str().expect("Material name must be a string"), material);
//                 }
//                 _ => panic!("Unknown material type: {}", material_type),
//             }
//         }
//         registry
//     }
// }