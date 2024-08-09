pub mod image;

use crate::types::color::Color;
use na::Point3;
use std::collections::BTreeMap;
use std::sync::Arc;

pub type TextureRef = Arc<dyn Texture + Send + Sync>;

pub trait Texture {
    fn value(&self, u: f32, v: f32, p: &Point3<f32>) -> Color;
}

pub struct TextureRegistry {
    textures: BTreeMap<String, TextureRef>,
}

impl TextureRegistry {
    pub fn new() -> Self {
        Self {
            textures: BTreeMap::new(),
        }
    }

    pub fn add_texture(&mut self, name: &str, texture: TextureRef) {
        self.textures.insert(name.to_string(), texture);
    }

    pub fn create_texture(
        &mut self,
        name: &str,
        texture: impl Texture + std::marker::Send + std::marker::Sync + 'static,
    ) {
        let texture: Arc<dyn Texture + Sync + Send> = Arc::new(texture);
        self.add_texture(name, texture);
    }

    pub fn get(&self, name: &str) -> Option<TextureRef> {
        match self.textures.get(name) {
            Some(texture) => Some(texture.clone()),
            None => {
                log::error!("Texture not found: {}", name);
                None
            }
        }
    }
}

pub struct Solid {
    albedo: Color,
}

impl Solid {
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Texture for Solid {
    fn value(&self, _u: f32, _v: f32, _p: &Point3<f32>) -> Color {
        self.albedo
    }
}

pub struct Checkered {
    odd: TextureRef,
    even: TextureRef,
    inv_scale: f32,
}

impl Checkered {
    pub fn new(odd: TextureRef, even: TextureRef, inv_scale: f32) -> Self {
        Self {
            odd,
            even,
            inv_scale,
        }
    }

    pub fn new_solid(odd: Color, even: Color, inv_scale: f32) -> Self {
        Self::new(
            Arc::new(Solid::new(odd)),
            Arc::new(Solid::new(even)),
            inv_scale,
        )
    }
}

impl Texture for Checkered {
    fn value(&self, u: f32, v: f32, p: &Point3<f32>) -> Color {
        // let x_int = (p.x * self.inv_scale).floor() as i64;
        // let y_int = (p.y * self.inv_scale).floor() as i64;
        // let z_int = (p.z * self.inv_scale).floor() as i64;

        let u_int = (u * self.inv_scale).floor() as i64;
        let v_int = (v * self.inv_scale).floor() as i64;

        if (u_int + v_int) % 2 == 0 {
            self.even.value(u, v, p)
        } else {
            self.odd.value(u, v, p)
        }
    }
}
