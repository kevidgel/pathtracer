pub mod lambertian;

use crate::types::{ray::Ray, color::Color};
use crate::objects::HitRecord;
use rand::rngs::ThreadRng;

pub trait Material {
    fn scatter(&self, rng: Option<&mut ThreadRng>, ray_in: &Ray, rec: &HitRecord) -> (Color, Ray);   
}