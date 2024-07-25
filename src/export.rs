use crate::types::color::Color;
use crate::types::color::ColorOps;

pub fn write_color(color: &Color) {
    let mut ir: u32 = (255.999 * color.get_r()) as u32;
    let mut ig: u32 = (255.999 * color.get_g()) as u32;
    let mut ib: u32 = (255.999 * color.get_b()) as u32;

    ir = ir.clamp(0, 255);
    ig = ig.clamp(0, 255);
    ib = ib.clamp(0, 255);

    println!("{} {} {}", ir, ig, ib);
}
