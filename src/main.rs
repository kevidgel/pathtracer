extern crate nalgebra as na;
mod config;
mod objects;
mod types;
mod camera;
mod materials;
mod textures;

use std::sync::Arc;

use na::Point3;
use objects::sphere::Sphere;
use objects::{Hittable, HittableObjects};
use camera::Camera;
use textures::{Checkered, TextureRegistry};
use types::color::Color;
use materials::{lambertian::Lambertian, metal::Metal, dielectric::Dielectric, MaterialRegistry};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    

    let aspect_ratio = 16_f32 / 9_f32;
    let image_width = 1980_u32;
    let vfov = 20_f32;

    let look_from = Point3::new(-2_f32, 2_f32, 1_f32);
    let look_at = Point3::new(0_f32, 0_f32, -1_f32);
    let focal_length = 1_f32;
    let defocus_angle = 0.0_f32;
    let spp = 256_u32;
    let camera = Camera::new(aspect_ratio, image_width, vfov, look_from, look_at, focal_length, defocus_angle, spp);

    log::info!(target: "pt", "Building scene...");
    
    // NOTE: These two will be shared by threads
    // TODO: after building, these will be read-only
    let mut materials = MaterialRegistry::new();
    let mut textures = TextureRegistry::new();
    let mut objects = HittableObjects::new();

    // TODO: We should be able to load this from a file (config.yml?)
    // Textures
    textures.create_texture("check_black", Checkered::new_solid(Color::new(0.1, 0.1, 0.1), Color::new(0.9, 0.9, 0.9), 26.0));

    // Materials
    materials.create_material("ground", Metal::new(Color::new(0.8, 0.8, 0.8), 0.1));
    materials.create_material("center", Lambertian::new_texture(textures.get("check_black")));
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
