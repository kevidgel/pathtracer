use crate::objects::PrimitiveBuffer;
use crate::types::sampler::{Sampler, SquareSampler};
use crate::types::{
    color::{Color, ColorOps},
    ray::Ray,
    sampler::DiskSampler,
};

use image::RgbImage;
use indicatif::{ProgressIterator, ProgressState, ProgressStyle};
use na::{Point3, Vector3};
use rand::rngs::ThreadRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::cmp;
use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

pub struct CameraConfig {
    pub aspect_ratio: f32,
    pub image_width: u32,
    pub vfov: f32,
    pub look_from: (f32, f32, f32),
    pub look_at: (f32, f32, f32),
    pub focal_length: f32,
    pub defocus_angle: f32,
    pub spp: u32,
    pub max_depth: u32,
    pub background: Color,
}

pub struct Camera {
    // Image params
    aspect_ratio: f32,
    vfov: f32,
    image_width: u32,
    image_height: u32,
    vup: Vector3<f32>, // Vertical up vector

    // Focus
    defocus_angle: f32,
    defocus_disk_u: Vector3<f32>,
    defocus_disk_v: Vector3<f32>,
    focal_length: f32,

    // Viewport
    center: Point3<f32>,
    pixel00: Point3<f32>,
    pixel_du: Vector3<f32>,
    pixel_dv: Vector3<f32>,

    // Tracing-related params
    spp: u32,
    sqrt_spp: u32,
    inv_sqrt_spp: f32,
    pixel_samples_scale: f32,
    max_depth: u32,
    background: Color,
}

impl Camera {
    pub fn from_config(cfg: &CameraConfig) -> Self {
        Self::new(
            cfg.aspect_ratio,
            cfg.image_width,
            cfg.vfov,
            Point3::new(cfg.look_from.0, cfg.look_from.1, cfg.look_from.2),
            Point3::new(cfg.look_at.0, cfg.look_at.1, cfg.look_at.2),
            cfg.focal_length,
            cfg.defocus_angle,
            cfg.spp,
            cfg.max_depth,
            cfg.background,
        )
    }

    pub fn new(
        aspect_ratio: f32,
        image_width: u32,
        vfov: f32,
        look_from: Point3<f32>,
        look_at: Point3<f32>,
        focal_length: f32,
        defocus_angle: f32,
        spp: u32,
        max_depth: u32,
        background: Color,
    ) -> Self {
        let image_height = cmp::max(1_u32, (image_width as f32 / aspect_ratio) as u32);
        let theta = vfov.to_radians();
        let h = (theta / 2_f32).tan();
        let vup = Vector3::new(0_f32, 1_f32, 0_f32);
        let center = look_from;

        let viewport_height = 2.0 * h * focal_length;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);

        let w = (look_from - look_at).normalize();
        let u = vup.cross(&w).normalize();
        let v = w.cross(&u);

        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;

        let pixel_du = viewport_u / (image_width as f32);
        let pixel_dv = viewport_v / (image_height as f32);

        let viewport_upper_left =
            center - viewport_u / 2_f32 - viewport_v / 2_f32 - focal_length * w;

        let pixel00 = viewport_upper_left + 0.5_f32 * pixel_du + 0.5_f32 * pixel_dv;

        let defocus_radius = focal_length * (0.5 * (defocus_angle / 2.0).to_radians()).tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        let sqrt_spp = (spp as f32).sqrt() as u32;
        let inv_sqrt_spp = 1.0 / sqrt_spp as f32;
        let pixel_samples_scale = 1.0 / ((sqrt_spp * sqrt_spp) as f32);

        Self {
            aspect_ratio,
            vfov,
            vup,
            image_width,
            image_height,
            defocus_angle,
            defocus_disk_u,
            defocus_disk_v,
            focal_length,
            center,
            pixel00,
            pixel_du,
            pixel_dv,
            spp,
            sqrt_spp,
            inv_sqrt_spp,
            pixel_samples_scale,
            max_depth,
            background,
        }
    }

    pub fn get_height(&self) -> u32 {
        self.image_height
    }

    pub fn get_width(&self) -> u32 {
        self.image_width
    }

    pub fn render(&self, objects: &PrimitiveBuffer, buffer: Arc<Mutex<RgbImage>>) {
        log::info!("Rendering...");
        let now = SystemTime::now();
        // Image generation
        // let mut buffer: RgbImage = ImageBuffer::new(self.image_width, self.image_height);
        let mut image: Vec<Color> =
            vec![Color::zeros(); self.image_width as usize * self.image_height as usize];

        // I'm too lazy to do it normally
        let mut sample_list = vec![];
        for s_u in 0..self.sqrt_spp {
            for s_v in 0..self.sqrt_spp {
                sample_list.push((s_u, s_v));
            }
        }

        let len = sample_list.len();
        let mut samples = 0;
        sample_list
            .into_iter()
            .progress_count(len as u64)
            .with_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% ({eta})",
                )
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                    write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                })
                .progress_chars("#>-"),
            )
            .for_each(|(s_u, s_v)| {
                samples += 1;
                image.par_iter_mut().enumerate().for_each(|(i, pixel)| {
                    let mut rng = rand::thread_rng();
                    let (u, v) = (
                        i % (self.image_width as usize),
                        i / (self.image_width as usize),
                    );
                    let ray = self.get_ray(&mut rng, u as f32, v as f32, s_u as f32, s_v as f32);
                    let pixel_color: Color =
                        self.ray_color(&mut rng, &ray, objects, self.max_depth); //* self.pixel_samples_scale;

                    *pixel += pixel_color;
                });

                let mut buffer = buffer.lock().unwrap();
                buffer.par_enumerate_pixels_mut().for_each(|(u, v, pixel)| {
                    let pixel_color: Color =
                        image[(u + v * self.image_width as u32) as usize] / (samples as f32);
                    *pixel = pixel_color.to_rgb();
                });
                //thread::sleep(Duration::from_millis(1));
            });

        let mut buffer = buffer.lock().unwrap();
        buffer.par_enumerate_pixels_mut().for_each(|(u, v, pixel)| {
            let pixel_color: Color =
                image[(u + v * self.image_width as u32) as usize] / (samples as f32);
            *pixel = pixel_color.to_rgb();
        });

        let render_elapsed = match now.elapsed() {
            Ok(elapsed) => elapsed,
            Err(e) => {
                log::error!("Failed to get elapsed time: {}", e);
                Duration::from_secs(0)
            }
        };

        log::info!("Render time: {:?}", render_elapsed);
    }

    fn get_ray(&self, rng: &mut ThreadRng, u: f32, v: f32, s_u: f32, s_v: f32) -> Ray {
        let sampler = SquareSampler::new((0.5, 0.5), 0.5);
        let disk_sampler = DiskSampler::unit();

        // ray_center
        let (disk_offset_u, disk_offset_v) = disk_sampler.sample(rng);
        let ray_center = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.center + disk_offset_u * self.defocus_disk_u + disk_offset_v * self.defocus_disk_v
        };

        // ray_direction
        let sample = sampler.sample(rng);
        let (offset_u, offset_v) = (
            (s_u + sample.0) * self.inv_sqrt_spp - 0.5,
            (s_v + sample.1) * self.inv_sqrt_spp - 0.5,
        );
        let pixel_center =
            self.pixel00 + (u + offset_u) * self.pixel_du + (v + offset_v) * self.pixel_dv;
        let ray_direction = pixel_center - ray_center;
        Ray::new(ray_center, ray_direction)
    }

    fn ray_color(
        &self,
        rng: &mut ThreadRng,
        ray: &Ray,
        objects: &PrimitiveBuffer,
        max_depth: u32,
    ) -> Color {
        if max_depth == 0 {
            return Color::zeros();
        }
        match objects.hit(ray, 0.001, f32::INFINITY) {
            Some(rec) => {
                match rec.material() {
                    Some(material) => {
                        let emitted = material.emitted(rec.u(), rec.v(), &rec.p());
                        match material.scatter(Some(rng), ray, &rec) {
                            Some((attenuation, scattered)) => {
                                let scattering_pdf = 1.0; //material.scattering_pdf(&ray, &scattered, &rec);
                                let pdf_value = scattering_pdf;
                                emitted
                                    + scattering_pdf
                                        * attenuation.component_mul(&self.ray_color(
                                            rng,
                                            &scattered,
                                            objects,
                                            max_depth - 1,
                                        ))
                                        / pdf_value
                            }
                            None => emitted,
                        }
                    }
                    // Unknown material
                    None => Color::zeros(),
                }
            }
            None => self.background,
        }
    }
}
