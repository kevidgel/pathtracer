extern crate nalgebra as na;

use image::Rgb;
pub type Color = na::Vector3<f32>;

pub trait ColorOps {
    fn get_r(&self) -> f32;
    fn get_g(&self) -> f32;
    fn get_b(&self) -> f32;
    fn to_u32(&self) -> u32;
    fn to_rgb(&self) -> Rgb<u8>;
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

    fn to_u32(&self) -> u32 {
        let mut ir: u32 = (255.999 * self.get_r()) as u32;
        let mut ig: u32 = (255.999 * self.get_g()) as u32;
        let mut ib: u32 = (255.999 * self.get_b()) as u32;

        ir = ir.clamp(0, 255);
        ig = ig.clamp(0, 255);
        ib = ib.clamp(0, 255);

        ir << 16 | ig << 8 | ib
    }

    fn to_rgb(&self) -> Rgb<u8> {
        let mut ir: u32 = (255.999 * self.get_r()) as u32;
        let mut ig: u32 = (255.999 * self.get_g()) as u32;
        let mut ib: u32 = (255.999 * self.get_b()) as u32;

        ir = ir.clamp(0, 255);
        ig = ig.clamp(0, 255);
        ib = ib.clamp(0, 255);

        Rgb([ir as u8, ig as u8, ib as u8])
    }
}
