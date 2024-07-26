use nalgebra::{Point3, Vector3};
use std::cmp;
use rayon::iter::ParallelIterator;
use crate::{types::{color::{Color, ColorOps}, ray::Ray}, Hittable, HittableObjects};
use image::{ImageBuffer, RgbImage};
use crate::types::sampler::{SquareSampler, Sampler};
use rand::rngs::ThreadRng;
use indicatif::ParallelProgressIterator;


pub struct Camera {
    aspect_ratio: f32,
    image_width: u32,
    image_height: u32,
    focal_length: f32,
    center: Point3<f32>,
    pixel00: Point3<f32>,
    pixel_du: Vector3<f32>,
    pixel_dv: Vector3<f32>,
    spp: u32,
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
        let spp: u32 = 32;

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
            spp,
        }
    }

    pub fn render(&self, objects: &HittableObjects) -> RgbImage {
        // Image generation
        let mut buffer: RgbImage = ImageBuffer::new(self.image_width, self.image_height);
        let len = buffer.len() as u64;
        // CPU parallelization
        buffer.par_enumerate_pixels_mut().progress_count(len).for_each(|(u, v, pixel)| {
            let mut rng = rand::thread_rng();
            let pixel_color: Color = (0..self.spp).map(|_| -> Color {
                let ray = self.get_ray(&mut rng, u as f32, v as f32);
                self.ray_color(&mut rng, &ray, objects, 8)
            }).sum::<Color>() / self.spp as f32;

            *pixel = pixel_color.to_rgb();
        });

        buffer
    }

    fn get_ray(&self, rng: &mut ThreadRng, u: f32, v: f32) -> Ray {
        let sampler = SquareSampler::new((0.0, 0.0), 0.5);
        let (offset_u, offset_v)= sampler.sample(rng);
        let pixel_center = self.pixel00 + (u + offset_u) * self.pixel_du + (v + offset_v) * self.pixel_dv;
        let ray_direction = pixel_center - self.center;
        Ray::new(self.center, ray_direction)
    }

    fn ray_color(&self, rng: &mut ThreadRng, ray: &Ray, objects: &HittableObjects, max_depth: u32) -> Color {
        if max_depth == 0 {
            return Color::zeros();
        }
        match objects.hit(ray, 0.001, f32::INFINITY) {
            Some(rec) => {
                match rec.material() {
                    Some(material) => {
                        let (throughput, next_ray) = material.scatter(Some(rng), ray, &rec);
                        throughput.component_mul(&self.ray_color(rng, &next_ray, objects, max_depth - 1))
                    }
                    // Unknown material
                    None => Color::zeros(),
                }
            }
            None => {
                let a = 0.5_f32 * (ray.direction.y + 1.0);
                (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
            }
        }
    }
}