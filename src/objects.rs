pub mod quad_mesh;
pub mod sphere;
pub mod tri_mesh;

use crate::bvh::BBox;
use crate::materials::MaterialRef;
use crate::types::ray::Ray;
use na::{Matrix, Matrix4, Point3, Vector3};
use std::sync::Arc;

pub type Primitive = Arc<dyn Hittable + Sync + Send>;

pub struct HitRecord {
    // Normal stuff
    p: Point3<f32>,
    normal: Vector3<f32>,
    t: f32,
    front_face: bool,

    // Material
    material: Option<MaterialRef>,

    // Texture
    u: f32,
    v: f32,
}

impl HitRecord {
    pub fn new(
        ray: &Ray,
        p: Point3<f32>,
        normal: Vector3<f32>,
        t: f32,
        material: Option<MaterialRef>,
        u: f32,
        v: f32,
    ) -> HitRecord {
        let mut rec: HitRecord = HitRecord {
            p,
            normal,
            t,
            front_face: false,
            material,
            u,
            v,
        };

        rec.set_face_normal(ray, normal);

        rec
    }

    pub fn p(&self) -> Point3<f32> {
        self.p
    }

    pub fn normal(&self) -> Vector3<f32> {
        self.normal
    }

    pub fn t(&self) -> f32 {
        self.t
    }

    pub fn front_face(&self) -> bool {
        self.front_face
    }

    pub fn material(&self) -> Option<MaterialRef> {
        self.material.clone()
    }

    pub fn u(&self) -> f32 {
        self.u
    }

    pub fn v(&self) -> f32 {
        self.v
    }

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector3<f32>) {
        // Flip the normal if the ray is inside the object
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

// We love traits !!!
pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    fn mat(&self) -> Option<MaterialRef>;
    fn bbox(&self) -> BBox;
}

// All of our things that are hittable
#[derive(Clone)]
pub struct HittableObjects {
    // TODO: replace with kd tree
    // TODO: idk if we should box this
    objs: Vec<Primitive>,
    bbox: BBox,
}

impl HittableObjects {
    pub fn new() -> HittableObjects {
        HittableObjects {
            objs: Vec::new(),
            bbox: BBox::empty(),
        }
    }

    pub fn objs_clone(&self) -> Vec<Primitive> {
        self.objs.clone()
    }

    pub fn add(&mut self, obj: Primitive) {
        self.bbox = self.bbox.merge(&obj.bbox());
        self.objs.push(obj);
    }

    pub fn add_all(&mut self, objs: Vec<Primitive>) {
        for obj in objs {
            self.add(obj);
        }
    }

    pub fn clear(&mut self) {
        self.objs.clear();
    }
}

impl Hittable for HittableObjects {
    // Ideally we replace this with spatial data structure
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for obj in &self.objs {
            if let Some(hit) = obj.hit(ray, t_min, closest_so_far) {
                closest_so_far = hit.t();
                rec = Some(hit);
            }
        }
        rec
    }

    fn mat(&self) -> Option<MaterialRef> {
        None
    }

    fn bbox(&self) -> BBox {
        self.bbox
    }
}

pub struct Instance {
    obj: Primitive,
    transform: Matrix4<f32>,
    inverse: Matrix4<f32>,
}

impl Instance {
    pub fn new(obj: Primitive, transform: na::Matrix4<f32>) -> Result<Self, &'static str> {
        let inverse = transform.try_inverse().ok_or("Matrix is not invertible")?;
        Ok(Instance {
            obj,
            transform,
            inverse,
        })
    }

    pub fn from_obj(obj: Primitive) -> Self {
        Instance {
            obj,
            transform: Matrix4::identity(),
            inverse: Matrix4::identity(),
        }
    }

    pub fn rotate(&mut self, angle: f32, axis: Vector3<f32>) {
        self.transform = Matrix4::new_rotation(angle * axis) * self.transform;
        self.inverse = self.inverse * Matrix4::new_rotation(-angle * axis);
    }

    pub fn rotate_x(&mut self, angle: f32) {
        self.transform = Matrix4::new_rotation(Vector3::new(angle, 0.0, 0.0)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_rotation(Vector3::new(-angle, 0.0, 0.0));
    }

    pub fn rotate_y(&mut self, angle: f32) {
        self.transform = Matrix4::new_rotation(Vector3::new(0.0, angle, 0.0)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_rotation(Vector3::new(0.0, -angle, 0.0));
    }

    pub fn rotate_z(&mut self, angle: f32) {
        self.transform = Matrix4::new_rotation(Vector3::new(0.0, 0.0, angle)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_rotation(Vector3::new(0.0, 0.0, -angle));
    }

    pub fn translate(&mut self, shift: Vector3<f32>) {
        self.transform = Matrix4::new_translation(&shift) * self.transform;
        self.inverse = self.inverse * Matrix4::new_translation(&(-shift));
    }

    pub fn translate_x(&mut self, x: f32) {
        self.transform = Matrix4::new_translation(&Vector3::new(x, 0.0, 0.0)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_translation(&Vector3::new(-x, 0.0, 0.0));
    }

    pub fn translate_y(&mut self, y: f32) {
        self.transform = Matrix4::new_translation(&Vector3::new(0.0, y, 0.0)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_translation(&Vector3::new(0.0, -y, 0.0));
    }

    pub fn translate_z(&mut self, z: f32) {
        self.transform = Matrix4::new_translation(&Vector3::new(0.0, 0.0, z)) * self.transform;
        self.inverse = self.inverse * Matrix4::new_translation(&Vector3::new(0.0, 0.0, -z));
    }

    pub fn scale(&mut self, scale: Vector3<f32>) {
        self.transform = Matrix4::new_nonuniform_scaling(&scale) * self.transform;
        self.inverse = self.inverse
            * Matrix4::new_nonuniform_scaling(&Vector3::new(
                1.0 / scale.x,
                1.0 / scale.y,
                1.0 / scale.z,
            ));
    }

    pub fn scale_x(&mut self, x: f32) {
        self.transform =
            Matrix4::new_nonuniform_scaling(&Vector3::new(x, 1.0, 1.0)) * self.transform;
        self.inverse =
            self.inverse * Matrix4::new_nonuniform_scaling(&Vector3::new(1.0 / x, 1.0, 1.0));
    }

    pub fn scale_y(&mut self, y: f32) {
        self.transform =
            Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, y, 1.0)) * self.transform;
        self.inverse =
            self.inverse * Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0 / y, 1.0));
    }

    pub fn scale_z(&mut self, z: f32) {
        self.transform =
            Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, z)) * self.transform;
        self.inverse =
            self.inverse * Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, 1.0 / z));
    }

    pub fn scale_uniform(&mut self, scale: f32) {
        self.transform = Matrix4::new_scaling(scale) * self.transform;
        self.inverse = self.inverse * Matrix4::new_scaling(1.0 / scale);
    }
}

impl Hittable for Instance {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let new_ray = Ray::new(
            self.inverse.transform_point(&ray.origin),
            self.inverse.transform_vector(&ray.direction),
        );
        match self.obj.hit(&new_ray, t_min, t_max) {
            Some(mut rec) => {
                rec.p = self.transform.transform_point(&rec.p);
                rec.normal = self.transform.transform_vector(&rec.normal);

                Some(rec)
            }
            None => None,
        }
    }

    fn mat(&self) -> Option<MaterialRef> {
        self.obj.mat()
    }

    fn bbox(&self) -> BBox {
        self.obj.bbox().transform(self.transform)
    }
}
