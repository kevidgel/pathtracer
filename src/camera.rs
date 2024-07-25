use nalgebra::{Point3, Vector3};
use std::cmp;
use crate::{HittableObjects, Hittable, types::ray::Ray, types::color::{Color, ColorOps}};
use image::{ImageBuffer, RgbImage};


pub struct Camera {
    aspect_ratio: f32,
    image_width: u32,
    image_height: u32,
    focal_length: f32,
    center: Point3<f32>,
    pixel00: Point3<f32>,
    pixel_du: Vector3<f32>,
    pixel_dv: Vector3<f32>,
}

impl Camera {
    pub fn new(aspect_ratio : f32, image_width: u32) -> Self {
        let image_height = cmp::max(1_u32, (image_width as f32 / aspect_ratio) as u32);
        let viewport_height = 2.0;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);
        let focal_length = 1.0;
        let center = Point3::new(0_f32, 0_f32, 0_f32);

        let viewport_u = Vector3::new(viewport_width, 0_f32, 0_f32);
        let viewport_v = Vector3::new(0_f32, -viewport_height, 0_f32);
        let pixel_du = viewport_u / (image_width as f32);
        let pixel_dv = viewport_v / (image_height as f32);

        let viewport_upper_left = center
            - viewport_u / 2_f32
            - viewport_v / 2_f32
            - Vector3::new(0_f32, 0_f32, focal_length);

        let pixel00 = viewport_upper_left + 0.5_f32 * pixel_du + 0.5_f32 * pixel_dv;

        Self {
            aspect_ratio,
            image_width,
            image_height,
            focal_length,
            center,
            pixel00,
            pixel_du,
            pixel_dv,
        }
    }

    pub fn render(&self, objects: &HittableObjects) -> RgbImage {
        // Image generation
        let mut buffer: RgbImage = ImageBuffer::new(self.image_width, self.image_height);

        buffer.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
            let pixel_center = self.pixel00 + (x as f32) * self.pixel_du + (y as f32) * self.pixel_dv;
            let ray_direction = pixel_center - self.center;
            let ray = Ray::new(self.center, ray_direction);

            let pixel_color = self.ray_color(&ray, objects);
            *pixel = pixel_color.to_rgb()
        });

        return buffer;
    }

    fn ray_color(&self, ray: &Ray, objects: &HittableObjects) -> Color {
        match objects.hit(ray, 0.0, f32::INFINITY) {
            Some(rec) => {
                let normal = rec.normal();
                return 0.5_f32 * Color::new(normal.x + 1_f32, normal.y + 1_f32, normal.z + 1_f32);
            }
            None => Color::zeros(),
        }
    }
}