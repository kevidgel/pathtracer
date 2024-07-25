extern crate nalgebra as na;
mod config;
mod export;
mod objects;
mod types;

use image::{ImageBuffer, Rgb, RgbImage};
use na::{Point3, Vector3};
use objects::sphere::Sphere;
use objects::Hittable;
use std::cmp;
use types::color::{Color, ColorOps};
use types::ray::Ray;

fn main() {
    env_logger::init();

    let cfg = match config::read_config("config.yml") {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!(target: "pt", "Config error: {}", e);
            return;
        }
    };

    // Image
    // TODO: Surely there is a better way to do this
    let aspect_ratio_vec: Vec<f32> = match cfg.get_array("aspect_ratio") {
        Ok(aspect_ratio) => aspect_ratio
            .iter()
            .map(|x| x.clone().into_float().unwrap_or(1_f64) as f32)
            .collect(),
        Err(_) => vec![1_f32, 1_f32],
    };

    let aspect_ratio: f32 = aspect_ratio_vec[0] / aspect_ratio_vec[1];
    let image_width: u32 = cfg.get_int("image_width").unwrap_or(200) as u32;
    let image_height: u32 = cmp::max(1_u32, (image_width as f32 / aspect_ratio) as u32);

    log::info!(target: "pt", "aspect ratio: {}, image_size: [{}, {}]", aspect_ratio, image_width, image_height);

    // Camera
    let focal_length: f32 = 1.0;
    let viewport_height: f32 = cfg.get_float("viewport_height").unwrap_or(2.0_f64) as f32;
    let viewport_width: f32 = viewport_height * (image_width as f32 / image_height as f32);
    let camera_center = Point3::new(0_f32, 0_f32, 0_f32);

    log::info!(target: "pt", "focal_length: {}, camera_center: [{}, {}, {}], viewport_size: [{}, {}]", focal_length, camera_center.x, camera_center.y, camera_center.z, viewport_width, viewport_height);

    // Viewport
    let viewport_u = Vector3::new(viewport_width, 0_f32, 0_f32);
    let viewport_v = Vector3::new(0_f32, -viewport_height, 0_f32);
    let pixel_delta_u = viewport_u / (image_width as f32);
    let pixel_delta_v = viewport_v / (image_height as f32);

    let viewport_upper_left = camera_center
        - viewport_u / 2_f32
        - viewport_v / 2_f32
        - Vector3::new(0_f32, 0_f32, focal_length);

    let pixel00_loc = viewport_upper_left + 0.5_f32 * pixel_delta_u + 0.5_f32 * pixel_delta_v;

    let test = Sphere::new(Point3::new(0_f32, 0_f32, -1_f32), 0.5_f32);

    // Ray coloring
    let ray_color = |ray: &Ray| -> Color {
        let mut rec = objects::HitRecord::new();
        if test.hit(ray, 0.0, f32::INFINITY, &mut rec) {
            return Color::new(
                0.5 * (rec.normal.x + 1.0),
                0.5 * (rec.normal.y + 1.0),
                0.5 * (rec.normal.z + 1.0),
            );
        } else {
            return Color::new(0.0, 0.0, 0.0);
        }
    };

    // Image generation
    let mut buffer: RgbImage = ImageBuffer::new(image_width, image_height);

    buffer.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let pixel_center = pixel00_loc + (x as f32) * pixel_delta_u + (y as f32) * pixel_delta_v;
        let ray_direction = pixel_center - camera_center;
        let ray = Ray::new(camera_center, ray_direction);

        let pixel_color = ray_color(&ray);
        *pixel = pixel_color.to_rgb()
    });

    buffer.save("test.png").unwrap();

    log::info!(target: "pt", "Done.");
}
