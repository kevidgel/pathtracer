use na::{Point3, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.origin + t * self.direction
    }

    pub fn set(&mut self, origin: Point3<f32>, direction: Vector3<f32>) {
        self.origin = origin;
        self.direction = direction;
    }
}
