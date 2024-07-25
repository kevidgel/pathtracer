extern crate nalgebra as na;

pub type Color = na::Vector3<f32>;

pub trait ColorOps {
    fn get_r(&self) -> f32;
    fn get_g(&self) -> f32;
    fn get_b(&self) -> f32;
}

impl ColorOps for Color {
    fn get_r(&self) -> f32 {
        self.x
    }

    fn get_g(&self) -> f32 {
        self.y
    }

    fn get_b(&self) -> f32 {
        self.z
    }
}
