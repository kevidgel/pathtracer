#![allow(dead_code)]

extern crate nalgebra as na;
mod bvh;
mod camera;
mod config;
mod materials;
mod objects;
mod scenes;
mod textures;
mod types;

use bvh::{BVHBuilder, SplitMethod};
use objects::Hittable;
use scenes::{cornell, lucy, Scene};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    log::info!("Building scene...");
    let now = std::time::SystemTime::now();
    let camera = cornell::Cornell::build_camera();
    let objects = cornell::Cornell::build_scene_flat_bvh();
    let objects = match objects {
        Ok(objects) => objects,
        _ => {
            log::error!("Failed to build BVH");
            return;
        }
    };

    let build_elapsed = match now.elapsed() {
        Ok(elapsed) => elapsed,
        Err(e) => {
            log::error!("Failed to get elapsed time: {}", e);
            std::time::Duration::from_secs(0)
        }
    };

    log::info!("Rendering...");
    let now = std::time::SystemTime::now();
    let buffer = camera.render(&objects);
    let render_elapsed = match now.elapsed() {
        Ok(elapsed) => elapsed,
        Err(e) => {
            log::error!("Failed to get elapsed time: {}", e);
            std::time::Duration::from_secs(0)
        }
    };

    buffer.save("test.png").unwrap();

    log::info!(
        "Done. Build time: {:?}. Render time: {:?}",
        build_elapsed,
        render_elapsed
    );
}
