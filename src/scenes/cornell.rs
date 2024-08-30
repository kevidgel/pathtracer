use na::{Point3, Vector3};

use super::Scene;
use crate::materials::dielectric::Dielectric;
use crate::objects::sphere::Sphere;
use crate::objects::{Instance, LightBuffer, PrimitiveBuffer};
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{lambertian::Lambertian, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::quad_mesh::Quad,
};

pub struct Cornell;

impl Scene for Cornell {
    fn build_camera() -> Camera {
        Camera::new(
            1.0,
            1024,
            40.0,
            Point3::new(278.0, 278.0, -800.0),
            Point3::new(278.0, 278.0, 0.0),
            1.0,
            0.0,
            128,
            16,
            Color::gray(0.0),
        )
    }

    fn build_scene() -> (PrimitiveBuffer, LightBuffer) {
        let mut objects = PrimitiveBuffer::new();
        let mut lights = LightBuffer::new();
        let mut materials = MaterialRegistry::new();

        materials.create_material("red", Lambertian::new(Color::new(0.65, 0.05, 0.05)));
        materials.create_material("mirror", Metal::new(Color::new(0.8, 0.85, 0.88), 0.05));
        materials.create_material("white", Lambertian::new(Color::gray(0.73)));
        materials.create_material("green", Lambertian::new(Color::new(0.12, 0.45, 0.15)));
        materials.create_material("light", Diffuse::new(Color::gray(15.0)));
        materials.create_material("glass", Dielectric::new(1.5_f32));

        let q1 = Quad::new(
            &Point3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("green"),
        );

        let q2 = Quad::new(
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("red"),
        );

        let q3 = Quad::new(
            &Point3::new(343.0, 554.0, 332.0),
            &Vector3::new(-130.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, -105.0),
            materials.get("light"),
        );

        let q4 = Quad::new(
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("white"),
        );

        let q5 = Quad::new(
            &Point3::new(555.0, 555.0, 555.0),
            &Vector3::new(-555.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, -555.0),
            materials.get("white"),
        );

        let q6 = Quad::new(
            &Point3::new(0.0, 0.0, 555.5),
            &Vector3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            materials.get("white"),
        );

        objects.add_quad(q1);
        objects.add_quad(q2);
        lights.add_quad(q3.clone());
        objects.add_quad(q3);
        objects.add_quad(q4);
        objects.add_quad(q5);
        objects.add_quad(q6);

        let box1 = Quad::new_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 330.0, 165.0),
            materials.get("mirror"),
        );

        let box2 = Quad::new_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 165.0, 165.0),
            materials.get("glass"),
        );

        let sphere1 = Sphere::new(
            Point3::new(190.0, 90.0, 190.0),
            90.0,
            materials.get("glass"),
        );

        let mut box1 = Instance::from_obj(box1);
        box1.rotate_y(15.0_f32.to_radians());
        box1.translate(Vector3::new(265.0, 0.0, 295.0));

        let mut box2 = Instance::from_obj(box2);
        box2.rotate_y(-18.0_f32.to_radians());
        box2.translate(Vector3::new(130.0, 0.0, 65.0));

        let mut sph = PrimitiveBuffer::new();

        sph.add_sphere(sphere1);

        let mut sph = Instance::from_obj(sph);

        objects.add_instance(box1);
        objects.add_instance(sph);

        (objects, lights)
    }
}
