use std::sync::Arc;

use na::{Point3, Vector3};

use super::Scene;
use crate::objects::{Instance, PrimitiveBuffer};
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{lambertian::Lambertian, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{quad_mesh::Quad, Primitive},
};

pub struct Cornell;

impl Scene<'_> for Cornell {
    fn build_camera() -> Camera {
        Camera::new(
            1.0,
            600,
            40.0,
            Point3::new(278.0, 278.0, -800.0),
            Point3::new(278.0, 278.0, 0.0),
            1.0,
            0.0,
            64,
            16,
            Color::gray(0.0),
        )
    }

    fn build_scene() -> PrimitiveBuffer {
        let mut objects = PrimitiveBuffer::new();
        let mut materials = MaterialRegistry::new();

        materials.create_material("red", Lambertian::new(Color::new(0.65, 0.05, 0.05)));
        materials.create_material("mirror", Metal::new(Color::new(0.73, 0.73, 0.73), 0.0));
        materials.create_material("white", Lambertian::new(Color::gray(0.73)));
        materials.create_material("green", Lambertian::new(Color::new(0.12, 0.45, 0.15)));
        materials.create_material("light", Diffuse::new(Color::gray(15.0)));

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

        objects.add(Primitive::Quad(q1));
        objects.add(Primitive::Quad(q2));
        objects.add(Primitive::Quad(q3));
        objects.add(Primitive::Quad(q4));
        objects.add(Primitive::Quad(q5));
        objects.add(Primitive::Quad(q6));

        let box1 = Quad::new_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 330.0, 165.0),
            materials.get("white"),
        );

        let box2 = Quad::new_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 165.0, 165.0),
            materials.get("white"),
        );

        let mut box1 = Instance::from_obj(box1);
        box1.rotate_y(15.0_f32.to_radians());
        box1.translate(Vector3::new(265.0, 0.0, 295.0));

        let mut box2 = Instance::from_obj(box2);
        box2.rotate_y(-18.0_f32.to_radians());
        box2.translate(Vector3::new(130.0, 0.0, 65.0));

        objects.add(Primitive::Instance(box1));
        objects.add(Primitive::Instance(box2));

        objects
    }
}
