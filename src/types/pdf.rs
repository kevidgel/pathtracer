use na::Vector3;
use rand::Rng;

use super::{color::ColorOps, onb::OrthonormalBasis};


pub trait PDF {
    fn value(&self, direction: &Vector3<f32>) -> f32;
    fn generate(&self, rng: &mut impl Rng) -> Vector3<f32>;
}

pub struct SpherePDF;

impl SpherePDF {
    pub fn new() -> Self {
        Self
    }
}

impl PDF for SpherePDF {
    fn value(&self, _direction: &Vector3<f32>) -> f32 {
        1.0 / (4.0 * std::f32::consts::PI)
    }

    fn generate(&self, rng: &mut impl Rng) -> Vector3<f32> {
        let (u, v) = (rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0));
        let norm_squared: f32 = u * u + v * v;
        
        if norm_squared <= 1.0 {
            let x = 2.0 * u * (1.0 - norm_squared).sqrt();
            let y = 2.0 * v * (1.0 - norm_squared).sqrt();
            let z = 1.0 - (2.0 * norm_squared);
            Vector3::new(x, y, z)
        } else {
            self.generate(rng)
        }
    }
}

pub struct CosineWeightedHemispherePDF {
    pub uvw: OrthonormalBasis,
}

impl CosineWeightedHemispherePDF {
    pub fn new(n: &Vector3<f32>) -> Self {
        Self {
            uvw: OrthonormalBasis::new(&n),
        }
    }
}

impl PDF for CosineWeightedHemispherePDF {
    fn value(&self, direction: &Vector3<f32>) -> f32 {
        let cos_t = direction.normalize().dot(&self.uvw.v());
        (cos_t / std::f32::consts::PI).max(0.0)
    }
    fn generate(&self, rng: &mut impl Rng) -> Vector3<f32> {
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        let out = self.uvw.to_world(&Vector3::new(x, y, z));
        assert!(self.value(&out) > 0.0);

        out
    }
}

pub struct HittablePDF {
    
}