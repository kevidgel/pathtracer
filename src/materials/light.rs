use super::Material;
use crate::objects::HitRecord;
use crate::textures::{Solid, TextureRef};
use crate::types::color::{Color, ColorOps};
use crate::types::ray::Ray;
use na::Point3;
use rand::rngs::ThreadRng;
use std::sync::Arc;

pub struct Diffuse {
    texture: TextureRef,
}

impl Diffuse {
    pub fn new(albedo: Color) -> Self {
        Self {
            texture: Arc::new(Solid::new(albedo)),
        }
    }

    pub fn gray(val: f32) -> Self {
        Self {
            texture: Arc::new(Solid::new(Color::gray(val))),
        }
    }

    pub fn new_texture(texture: Option<TextureRef>) -> Self {
        Self {
            texture: texture.unwrap_or(Arc::new(Solid::new(Color::gray(0.5)))),
        }
    }
}

impl Material for Diffuse {
    fn is_emissive(&self) -> bool {
        true
    }

    fn is_specular(&self) -> bool {
        true
    }
    fn scatter(&self, _rng: &mut ThreadRng, _ray_in: &Ray, _rec: &HitRecord) -> Option<Ray> {
        None
    }

    fn emitted(&self, _ray: &Ray, rec: &HitRecord, u: f32, v: f32, p: &Point3<f32>) -> Color {
        if !rec.front_face() {
            return Color::zeros();
        }
        self.texture.value(u, v, p)
    }
}
