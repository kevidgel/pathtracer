use super::Texture;
use crate::types::color::{Color, ColorOps};
use image::{ImageBuffer, RgbImage};
use na::{Point3, Vector3};

pub struct Image {
    image: Option<RgbImage>,
}

impl Image {
    pub fn new(image: Option<RgbImage>) -> Self {
        Self { image }
    }

    pub fn load(path: &str) -> Self {
        Self {
            image: match image::open(path) {
                Ok(image) => Some(image.to_rgb8()),
                Err(_) => {
                    log::error!(
                        "Failed to load image: '{}'... Using default texture instead...",
                        path
                    );
                    None
                }
            },
        }
    }
}

impl Texture for Image {
    fn value(&self, u: f32, v: f32, _p: &Point3<f32>) -> Color {
        match &self.image {
            Some(image) => {
                if (image.height() <= 0) || (image.width() <= 0) {
                    Color::gray(0.0)
                } else {
                    let u = u.clamp(0.0, 1.0);
                    let v = 1.0 - v.clamp(0.0, 1.0);

                    let (i, j) = (
                        (u * image.width() as f32) as u32,
                        (v * image.height() as f32) as u32,
                    );

                    let pixel = image.get_pixel(i, j);
                    Color::new(
                        pixel[0] as f32 / 255.0,
                        pixel[1] as f32 / 255.0,
                        pixel[2] as f32 / 255.0,
                    )
                }
            }
            None => {
                let u_int = (u * 64.0).floor() as i64;
                let v_int = (v * 64.0).floor() as i64;

                if (u_int + v_int) % 2 == 0 {
                    Color::gray(1.0)
                } else {
                    Color::gray(0.7)
                }
            }
        }
    }
}
