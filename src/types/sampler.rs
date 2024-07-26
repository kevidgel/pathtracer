use rand::Rng;
use nalgebra::Vector3;

// TODO: Maybe parameterize this trait
pub trait Sampler<T> {
    fn sample(&self, rng: &mut impl Rng) -> T;
}

pub struct SquareSampler {
    center: (f32, f32),
    apothem: f32,
}

impl SquareSampler {
    pub fn new(center: (f32, f32), apothem: f32) -> Self {
        Self { center, apothem }
    }
}

impl Sampler<(f32, f32)> for SquareSampler {
    fn sample(&self, rng: &mut impl Rng) -> (f32, f32) {
        let x = rng.gen_range(self.center.0 - self.apothem..self.center.0 + self.apothem);
        let y = rng.gen_range(self.center.1 - self.apothem..self.center.1 + self.apothem);

        (x, y)
    }
}

pub struct SphereSampler {
    center: Vector3<f32>,
    radius: f32,
}

impl SphereSampler {
    pub fn new(center: Vector3<f32>, radius: f32) -> Self {
        Self { center, radius }
    }
    pub fn unit() -> Self {
        Self { center: Vector3::new(0.0, 0.0, 0.0), radius: 1.0 }
    }
}

impl Sampler<Vector3<f32>> for SphereSampler {
    fn sample(&self, rng: &mut impl Rng) -> Vector3<f32> {
        // Rejection sampling
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        let z = rng.gen_range(-1.0..1.0);

        let sample: Vector3<f32> = Vector3::new(x, y, z);
        if sample.norm() < 1.0 {
            sample.normalize()
        } else {
            self.sample(rng)
        }
    }    
}