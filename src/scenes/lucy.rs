use std::sync::Arc;

use na::Point3;

use super::Scene;
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{dielectric::Dielectric, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{sphere::Sphere, tri_mesh::TriMesh, HittableObjects},
};

pub struct Lucy;

impl Scene for Lucy {
    fn build_scene() -> HittableObjects {
        let mut objects = HittableObjects::new();
        let mut materials = MaterialRegistry::new();
        let meshes: Vec<TriMesh> = TriMesh::load_as_vec("lucy.obj");

        materials.create_material("ground", Metal::new(Color::new(0.3, 0.3, 0.8), 0.1));
        materials.create_material("emit", Diffuse::gray(1.0));
        materials.create_material("mat1", Dielectric::new(1.5_f32));

        let ground = Sphere::new(
            Point3::new(0_f32, -1000.4_f32, 0_f32),
            1000_f32,
            materials.get("ground"),
        );

        let light = Sphere::new(Point3::new(0.7, 2.3, 1.0), 0.9, materials.get("emit"));

        objects.add(Arc::new(light));
        objects.add(Arc::new(ground));

        let mesh = meshes.get(0).unwrap();
        let triangles = mesh.to_triangles_with_mat(materials.get("mat1").unwrap());
        for tri in triangles {
            objects.add(Arc::new(tri));
        }

        objects
    }

    fn build_camera() -> Camera {
        let aspect_ratio = 16_f32 / 9_f32;
        let image_width = 1200_u32;
        let vfov = 20_f32;

        let look_from = Point3::new(-3_f32, 2_f32, 5_f32);
        let look_at = Point3::new(0_f32, 0.4_f32, 0_f32);
        let focal_length = 10_f32;
        let defocus_angle = 0_f32;
        let spp = 512_u32;
        let max_depth = 16_u32;
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
            Color::gray(0.5),
        );

        camera
    }
}
