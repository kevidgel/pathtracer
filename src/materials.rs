pub mod lambertian;

use crate::types::{ray::Ray, color::Color};
use crate::objects::HitRecord;
use rand::Rng;

pub trait Material {
    fn scatter(&self, rng: Option<&mut impl Rng>, ray_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)>;   
}