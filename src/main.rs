extern crate nalgebra as na;
mod bvh;
mod camera;
mod config;
mod materials;
mod objects;
mod textures;
mod types;

use std::sync::Arc;

use bvh::{BVHBuilder, SplitMethod};
use camera::Camera;
use materials::{dielectric::Dielectric, lambertian::Lambertian, metal::Metal, MaterialRegistry};
use na::{Point3, Vector3};
use objects::sphere::Sphere;
use objects::tri_mesh::{TriMesh, Triangle};
use objects::{Hittable, HittableObjects};
use rand::Rng;
use textures::{image::Image, Checkered, TextureRegistry};
use types::color::{Color, ColorOps};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let aspect_ratio = 16_f32 / 9_f32;
    let image_width = 1920_u32;
    let vfov = 20_f32;

    let look_from = Point3::new(-3_f32, 2_f32, 5_f32);
    let look_at = Point3::new(0_f32, 0_f32, 0_f32);
    let focal_length = 10_f32;
    let defocus_angle = 0_f32;
    let spp = 512_u32;
    let max_depth = 32_u32;
    let camera = Camera::new(
        aspect_ratio,
        image_width,
        vfov,
        look_from,
        look_at,
        focal_length,
        defocus_angle,
        spp,
        max_depth,
    );

    log::info!("Building scene...");

    // NOTE: These two will be shared by threads
    // TODO: after building, these will be read-only
    let mut materials = MaterialRegistry::new();
    let mut textures = TextureRegistry::new();
    let mut objects = HittableObjects::new();

    // TODO: We should be able to load this from a file (config.yml?)
    // Textures
    textures.create_texture(
        "check_black",
        Checkered::new_solid(Color::new(0.1, 0.1, 0.1), Color::new(0.9, 0.9, 0.9), 26.0),
    );
    textures.create_texture("spot", Image::load("spot_texture.png"));
    // Materials
    materials.create_material("ground", Metal::new(Color::new(0.3, 0.3, 0.8), 0.1));
    materials.create_material("left", Dielectric::new(1.5_f32));
    materials.create_material("bubble", Dielectric::new(1.0_f32 / 1.5_f32));
    materials.create_material("right", Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));

    // Objects
    let ground = Sphere::new(
        Point3::new(0_f32, -1000.4_f32, 0_f32),
        1000_f32,
        materials.get("ground"),
    );
    // let center = Sphere::new(
    //     Point3::new(0_f32, 0_f32, -1.2_f32),
    //     0.5,
    //     materials.get("center"),
    // );
    // let left = Sphere::new(
    //     Point3::new(-1_f32, 0_f32, -1_f32),
    //     0.5,
    //     materials.get("left"),
    // );
    // let bubble = Sphere::new(
    //     Point3::new(-1_f32, 0_f32, -1_f32),
    //     0.4,
    //     materials.get("bubble"),
    // );
    // let right = Sphere::new(
    //     Point3::new(1_f32, 0_f32, -1_f32),
    //     0.5,
    //     materials.get("right"),
    // );

    // let mut rng = rand::thread_rng();
    // for i in -11..11 {
    //     for j in -11..11 {
    //         let choose_mat: f32 = rng.gen();
    //         let center = Point3::new(
    //             i as f32 + 0.9 * rng.gen::<f32>(),
    //             0.2,
    //             j as f32 + 0.9 * rng.gen::<f32>(),
    //         );

    //         if (center - Point3::new(4.0, 0.2, 0.0)).norm() > 0.9 {
    //             if choose_mat < 0.8 {
    //                 // diffuse
    //                 let albedo = Color::random().component_mul(&ColorOps::random());
    //                 let sphere = Sphere::new(center, 0.2, Some(Arc::new(Lambertian::new(albedo))));
    //                 objects.add(Arc::new(sphere));
    //             } else if choose_mat < 0.95 {
    //                 // metal
    //                 let albedo = Color::random_range(0.5, 1.0);
    //                 let fuzz = rng.gen_range(0.0..0.5);
    //                 let sphere = Sphere::new(center, 0.2, Some(Arc::new(Metal::new(albedo, fuzz))));
    //                 objects.add(Arc::new(sphere));
    //             } else {
    //                 // glass
    //                 let sphere = Sphere::new(center, 0.2, Some(Arc::new(Dielectric::new(1.5))));
    //                 objects.add(Arc::new(sphere));
    //             }
    //         }
    //     }
    // }

    materials.create_material("mat1", Dielectric::new(1.5_f32));
    materials.create_material("mat2", Lambertian::new(Color::new(0.8, 0.8, 0.8)));
    materials.create_material("mat3", Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));

    // objects.add(Arc::new(Sphere::new(
    //     Point3::new(0.0, 1.0, 0.0),
    //     1.0,
    //     materials.get("mat1"),
    // )));
    // objects.add(Arc::new(Sphere::new(
    //     Point3::new(-4.0, 1.0, 0.0),
    //     1.0,
    //     materials.get("mat2"),
    // )));
    // objects.add(Arc::new(Sphere::new(
    //     Point3::new(4.0, 1.0, 0.0),
    //     1.0,
    //     materials.get("mat3"),
    // )));

    objects.add(Arc::new(ground));
    // objects.add(Arc::new(center));
    // objects.add(Arc::new(left));
    // objects.add(Arc::new(bubble));
    // objects.add(Arc::new(right));

    let meshes = TriMesh::load_as_vec("dragon.obj");

    let mesh = meshes.get(0).unwrap();
    materials.add_material(
        "spot",
        Arc::new(Lambertian::new_texture(textures.get("spot"))),
    );
    let triangles = mesh.to_triangles_with_mat(materials.get("mat2").unwrap());

    for tri in triangles {
        objects.add(Arc::new(tri));
    }

    let objects = BVHBuilder::build_flattened_from_hittable_objects(SplitMethod::Middle, objects);

    log::info!("Rendering...");
    let buffer = camera.render(&objects);

    buffer.save("test.png").unwrap();

    log::info!("Done.");
}
