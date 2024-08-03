use super::Material;
use crate::textures::{TextureRef, Solid};
use crate::types::color::{Color, ColorOps};
use std::sync::Arc;
use na::Point3;
use crate::types::ray::Ray;
use rand::rngs::ThreadRng;
use crate::objects::HitRecord;

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
    fn scatter(&self, _rng: Option<&mut ThreadRng>, _ray_in: &Ray, _rec: &HitRecord) -> Option<(Color, Ray)> {
        None
    }

    fn emitted(&self, u: f32, v: f32, p: &Point3<f32>) -> Color {
        self.texture.value(u, v, p)
    }
}
