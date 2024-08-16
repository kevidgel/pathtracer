use std::sync::Arc;

use na::{Point3, Vector3};

use super::Scene;
use crate::objects::Instance;
use crate::types::color::{Color, ColorOps};
use crate::{
    camera::Camera,
    materials::{lambertian::Lambertian, light::Diffuse, metal::Metal, MaterialRegistry},
    objects::{quad_mesh::Quad, HittableObjects, Primitive},
};

pub struct Cornell;

impl Scene for Cornell {
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
            50,
            Color::gray(0.0),
        )
    }

    fn build_scene() -> HittableObjects {
        let mut objects = HittableObjects::new();
        let mut materials = MaterialRegistry::new();

        materials.create_material("red", Lambertian::new(Color::new(0.65, 0.05, 0.05)));
        materials.create_material("mirror", Metal::new(Color::new(0.73, 0.73, 0.73), 0.0));
        materials.create_material("white", Lambertian::new(Color::gray(0.73)));
        materials.create_material("green", Lambertian::new(Color::new(0.12, 0.45, 0.15)));
        materials.create_material("light", Diffuse::new(Color::gray(15.0)));

        let q1 = Arc::new(Quad::new(
            &Point3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("green"),
        ));

        let q2 = Arc::new(Quad::new(
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("red"),
        ));

        let q3 = Arc::new(Quad::new(
            &Point3::new(343.0, 554.0, 332.0),
            &Vector3::new(-130.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, -105.0),
            materials.get("light"),
        ));

        let q4 = Arc::new(Quad::new(
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 555.0),
            materials.get("white"),
        ));

        let q5 = Arc::new(Quad::new(
            &Point3::new(555.0, 555.0, 555.0),
            &Vector3::new(-555.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, -555.0),
            materials.get("white"),
        ));

        let q6 = Arc::new(Quad::new(
            &Point3::new(0.0, 0.0, 555.5),
            &Vector3::new(555.0, 0.0, 0.0),
            &Vector3::new(0.0, 555.0, 0.0),
            materials.get("white"),
        ));

        objects.add_all(q1.to_triangles());
        objects.add_all(q2.to_triangles());
        objects.add_all(q3.to_triangles());
        objects.add_all(q4.to_triangles());
        objects.add_all(q5.to_triangles());
        objects.add_all(q6.to_triangles());

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

        let mut box1 = Instance::from_obj(Arc::new(box1) as Primitive);
        box1.rotate_y(15.0_f32.to_radians());
        box1.translate(Vector3::new(265.0, 0.0, 295.0));

        let mut box2 = Instance::from_obj(Arc::new(box2) as Primitive);
        box2.rotate_y(-18.0_f32.to_radians());
        box2.translate(Vector3::new(130.0, 0.0, 65.0));

        objects.add(Arc::new(box1));
        objects.add(Arc::new(box2));

        objects
    }
}
