use na::Point3;

use super::Scene;
use crate::materials::lambertian::{self, Lambertian};
use crate::objects::{LightBuffer, PrimitiveBuffer};
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{dielectric::Dielectric, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{sphere::Sphere, tri_mesh::TriMesh},
};

pub struct Lucy;

impl Scene for Lucy {
    fn build_scene() -> (PrimitiveBuffer, LightBuffer) {
        let mut objects = PrimitiveBuffer::new();
        let mut lights = LightBuffer::new();
        let mut materials = MaterialRegistry::new();
        let meshes: Vec<TriMesh> = TriMesh::load_as_vec("/home/kevidgel/Downloads/sponza/sponza_tri.obj");

        materials.create_material("ground", Metal::new(Color::new(0.3, 0.3, 0.8), 0.1));
        materials.create_material("emit", Diffuse::new(Color::new(255.0, 255.0, 255.0)));
        materials.create_material("mat1", Dielectric::new(1.5_f32));
        materials.create_material("mat2", Lambertian::new(Color::new(0.5, 0.5, 0.5)));

        // let ground = Sphere::new(
        //     Point3::new(0_f32, -1000.4_f32, 0_f32),
        //     1000_f32,
        //     materials.get("ground"),
        // );

        let light = Sphere::new(Point3::new(0.7, 2.3, 1.0), 0.5, materials.get("emit"));

        lights.add_sphere(light.clone());
        objects.add_sphere(light);
        //objects.add_sphere(ground);

        for mesh in meshes {
            let triangles = mesh.to_triangles_with_mat(materials.get("mat2").unwrap());
            for tri in triangles {
                objects.add_triangle(tri);
            }
        }

        (objects, lights)
    }

    fn build_camera() -> Camera {
        let aspect_ratio = 16_f32 / 9_f32;
        let image_width = 1200_u32;
        let vfov = 20_f32;

        let look_from = Point3::new(-12_f32, 18_f32, 0_f32);
        let look_at = Point3::new(20_f32, 0.4_f32, 0_f32);
        let focal_length = 10_f32;
        let defocus_angle = 0_f32;
        let spp = 51200_u32;
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
            Color::gray(1.5),
        );

        camera
    }
}
