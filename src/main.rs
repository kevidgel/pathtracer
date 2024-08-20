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

use eframe::egui;
use image::RgbImage;
use std::sync::mpsc;


use scenes::{cornell, lucy, Scene};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

struct PathtracerApp {
    image_buffer: Arc<Mutex<RgbImage>>,
    rx: mpsc::Receiver<()>,
    texture: Option<egui::TextureHandle>,
    width: usize,
    height: usize,
}

impl eframe::App for PathtracerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.rx.try_recv().is_ok() {
            log::info!("Done");
        }
        egui::CentralPanel::default().frame(egui::containers::Frame::none()).show(ctx, |ui| {
            let buffer = self.image_buffer.lock().unwrap();

            if self.texture.is_none() {
                self.texture = Some(ui.ctx().load_texture(
                    "raytraced_image",
                    egui::ColorImage::from_rgb([self.width, self.height], &buffer),
                    egui::TextureOptions::default(),
                ));
            } else {
                self.texture.as_mut().unwrap().set(
                    egui::ColorImage::from_rgb([self.width, self.height], &buffer),
                    egui::TextureOptions::default(),
                );
            }

            if let Some(texture) = &self.texture {
                ui.image(texture);
            }
        });

        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Building scene...");
    let now = SystemTime::now();
    let camera = cornell::Cornell::build_camera();
    let (width, height) = (camera.get_width() as usize, camera.get_height() as usize);
    let mut objects = cornell::Cornell::build_scene();
    objects.build_bvh();

    let build_elapsed = match now.elapsed() {
        Ok(elapsed) => elapsed,
        Err(e) => {
            log::error!("Failed to get elapsed time: {}", e);
            Duration::from_secs(0)
        }
    };

    let image_buffer: Arc<Mutex<RgbImage>> = Arc::new(Mutex::new(RgbImage::new(
        width as u32,
        height as u32,
    )));

    log::info!("Build time: {:?}", build_elapsed);

    let image_buffer_to_render = image_buffer.clone();

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let to_save = image_buffer_to_render.clone();
        camera.render(&objects, image_buffer_to_render);
        to_save.lock().unwrap().save("strat.png").unwrap();
        tx.send(()).unwrap();
    });

    let app = PathtracerApp {
        image_buffer: image_buffer.clone(),
        rx,
        texture: None,
        width,
        height,
    };

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder {
            inner_size: Some(egui::Vec2::new(width as f32, height as f32)),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Pathtracer",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
