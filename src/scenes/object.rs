use na::{Point3, Vector3};

use super::Scene;
use crate::materials::lambertian::Lambertian;
use crate::materials::disney::{DisneyClearcoat, DisneyDiffuse, DisneyMetal, DisneySheen};
use crate::objects::quad_mesh::Quad;
use crate::objects::{LightBuffer, PrimitiveBuffer};
use crate::textures::{Checkered, Solid};
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{dielectric::Dielectric, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{sphere::Sphere, tri_mesh::TriMesh},
};
use std::sync::Arc;

pub struct Object;

impl Scene for Object {
    fn build_scene() -> (PrimitiveBuffer, LightBuffer) {
        let mut objects = PrimitiveBuffer::new();
        let mut lights = LightBuffer::new();
        let mut materials = MaterialRegistry::new();
        let meshes: Vec<TriMesh> = TriMesh::load_as_vec("mitsuba_smooth.obj");

        let lite = Arc::new(Solid::new(Color::gray(0.6)));
        let dark = Arc::new(Solid::new(Color::gray(0.2)));
        let check = Arc::new(Checkered::new(lite, dark, 19.0));

        materials.create_material("ground", Metal::new(Color::gray(0.45), 0.5));
        materials.create_material("emit", Diffuse::new(Color::gray(20.0)));
        materials.create_material("check", Lambertian::new_texture(Some(check)));
        materials.create_material("mat1", Lambertian::new(Color::gray(0.2)));
        materials.create_material("mat2", Lambertian::new(Color::new(0.3, 0.3, 0.8)));

        let sheen = DisneySheen::new(Color::new(0.3, 0.3, 0.8), 1.0, 0.2);
        let diffuse = DisneyDiffuse::new(Color::new(0.3, 0.3, 0.8), 1.0, 0.5);
        let clearcoat = DisneyClearcoat::new(1.0, 0.8);
        let metal = DisneyMetal::new(Color::new(0.3, 0.3, 0.8), 0.5, 0.3);
        materials.create_material("test", metal);
        // let ground = Sphere::new(
        //     Point3::new(0_f32, -1000.4_f32, 0_f32),
        //     1000_f32,
        //     materials.get("ground"),
        // );
        // let ground = Quad::new(&Point3::new(-1000.0, -1.0, -1000.0), &Vector3::new(2000.0, 0.0, 0.0), &Vector3::new(0.0, 0.0, 2000.0), materials.get("mat2"));

        let light = Sphere::new(Point3::new(-10.0, 15.0, 5.0), 10.0, materials.get("emit"));

        lights.add_sphere(light.clone());
        objects.add_sphere(light);
        // objects.add_quad(ground);
        let mats = vec!["check", "mat1", "test"];

        for i in 0..meshes.len() {
            let triangles = meshes[i].to_triangles_with_mat(materials.get(mats[i]).unwrap());
            for tri in triangles {
                objects.add_triangle(tri);
            }
        }

        (objects, lights)
    }

    fn build_camera() -> Camera {
        let aspect_ratio = 1.0;
        let image_width = 768_u32;
        let vfov = 15_f32;

        let look_from = Point3::new(2_f32, 10_f32, 16_f32);
        let look_at = Point3::new(0_f32, 1.0_f32, 0_f32);
        let focal_length = 10_f32;
        let defocus_angle = 0_f32;
        let spp = 10240_u32;
        let max_depth = 50_u32;
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
            Color::gray(0.0),
        );

        camera
    }
}
