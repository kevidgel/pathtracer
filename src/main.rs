extern crate nalgebra as na;
mod config;
mod export;
mod objects;
mod types;
mod camera;
mod materials;

use std::sync::Arc;

use na::Point3;
use objects::sphere::Sphere;
use objects::{Hittable, HittableObjects};
use camera::Camera;

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

    let aspect_ratio = aspect_ratio_vec[0] / aspect_ratio_vec[1];
    let image_width: u32 = cfg.get_int("image_width").unwrap_or(200) as u32;
    
    let camera = Camera::new(aspect_ratio, image_width);

    log::info!(target: "pt", "Building scene...");
    let mut objects = HittableObjects::new();

    let test1 = Sphere::new(Point3::new(0_f32, 0_f32, -1_f32), 0.5_f32);
    let test2 = Sphere::new(Point3::new(0_f32, -100.5_f32, -1_f32), 100_f32);

    objects.add(Arc::new(test1));
    objects.add(Arc::new(test2));

    log::info!(target: "pt", "Rendering...");
    let buffer = camera.render(&objects);

    buffer.save("test.png").unwrap();

    log::info!(target: "pt", "Done.");
}
