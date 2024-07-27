extern crate nalgebra as na;
mod config;
mod objects;
mod types;
mod camera;
mod materials;

use std::sync::Arc;

use na::Point3;
use objects::sphere::Sphere;
use objects::{Hittable, HittableObjects};
use camera::Camera;
use types::color::Color;
use materials::{lambertian::Lambertian, metal::Metal, dielectric::Dielectric, MaterialRegistry};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

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
    let vfov: f32 = cfg.get_float("vfov").unwrap_or(20_f64) as f32;
    
    let look_from = Point3::new(-2_f32, 2_f32, 1_f32);
    let look_at = Point3::new(0_f32, 0_f32, -1_f32);
    let camera = Camera::new(aspect_ratio, image_width, vfov, look_from, look_at, 3.4, 10.0,256);

    log::info!(target: "pt", "Building scene...");
    
    // NOTE: These two will be shared by threads
    // TODO: after building, these will be read-only
    let mut materials = MaterialRegistry::new();
    let mut objects = HittableObjects::new();

    // TODO: We should be able to load this from a file (config.yml?)

    // Materials
    materials.create_material("ground", Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    materials.create_material("center", Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    materials.create_material("left", Dielectric::new(1.5_f32));
    materials.create_material("bubble", Dielectric::new(1.0_f32/ 1.5_f32));
    materials.create_material("right", Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));
    
    // Objects
    let ground = Sphere::new(Point3::new(0_f32, -100.5_f32, -1_f32), 100_f32, materials.get("ground"));
    let center = Sphere::new(Point3::new(0_f32, 0_f32, -1.2_f32), 0.5, materials.get("center"));
    let left = Sphere::new(Point3::new(-1_f32, 0_f32, -1_f32), 0.5, materials.get("left"));
    let bubble = Sphere::new(Point3::new(-1_f32, 0_f32, -1_f32), 0.4, materials.get("bubble"));
    let right = Sphere::new(Point3::new(1_f32, 0_f32, -1_f32), 0.5, materials.get("right"));

    objects.add(Arc::new(ground));
    objects.add(Arc::new(center));
    objects.add(Arc::new(left));
    objects.add(Arc::new(bubble));
    objects.add(Arc::new(right));

    log::info!(target: "pt", "Rendering...");
    let buffer = camera.render(&objects);

    buffer.save("test.png").unwrap();

    log::info!(target: "pt", "Done.");
}
