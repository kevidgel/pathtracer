use na::{Point3, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub t_min: f32,
    pub t_max: f32,
}

impl Ray {
    pub fn new_bounded(origin: Point3<f32>, direction: Vector3<f32>, t_min: f32, t_max: f32) -> Self {
        Self { origin, direction, t_min, t_max }
    }

    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self::new_bounded(origin, direction, 0.001, f32::INFINITY)
    }

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.origin + t * self.direction
    }

    pub fn set(&mut self, origin: Point3<f32>, direction: Vector3<f32>) {
        self.origin = origin;
        self.direction = direction;
    }
}
