use na::{Point3, Vector3};

use super::Scene;
use crate::materials::lambertian::Lambertian;
use crate::objects::quad_mesh::Quad;
use crate::objects::{LightBuffer, PrimitiveBuffer};
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{dielectric::Dielectric, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{sphere::Sphere, tri_mesh::TriMesh},
};

pub struct Object;

impl Scene for Object {
    fn build_scene() -> (PrimitiveBuffer, LightBuffer) {
        let mut objects = PrimitiveBuffer::new();
        let mut lights = LightBuffer::new();
        let mut materials = MaterialRegistry::new();
        let meshes: Vec<TriMesh> = TriMesh::load_as_vec("buddha.obj");

        materials.create_material("ground", Metal::new(Color::new(0.3, 0.3, 0.8), 0.5));
        materials.create_material("emit", Diffuse::new(Color::new(10.0, 10.0, 10.0)));
        materials.create_material("mat1", Lambertian::new(Color::gray(0.5)));
        materials.create_material("mat2", Lambertian::new(Color::new(0.2, 0.2, 0.8)));

        // let ground = Sphere::new(
        //     Point3::new(0_f32, -1000.4_f32, 0_f32),
        //     1000_f32,
        //     materials.get("ground"),
        // );
        // let ground = Quad::new(&Point3::new(-1000.0, -1.0, -1000.0), &Vector3::new(2000.0, 0.0, 0.0), &Vector3::new(0.0, 0.0, 2000.0), materials.get("mat2"));

        let light = Sphere::new(Point3::new(0.0, 5.0, -1.0), 3.0, materials.get("emit"));

        lights.add_sphere(light.clone());
        objects.add_sphere(light);
        // objects.add_quad(ground);

        for mesh in meshes {
            let triangles = mesh.to_triangles_with_mat(materials.get("mat1").unwrap());
            for tri in triangles {
                objects.add_triangle(tri);
            }
        }

        (objects, lights)
    }

    fn build_camera() -> Camera {
        let aspect_ratio = 1.0;
        let image_width = 1024_u32;
        let vfov = 20_f32;

        let look_from = Point3::new(0_f32, 1_f32, -4_f32);
        let look_at = Point3::new(0_f32, 0_f32, 0_f32);
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
            Color::gray(0.01),
        );

        camera
    }
}
