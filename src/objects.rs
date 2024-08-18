pub mod quad_mesh;
pub mod sphere;
pub mod tri_mesh;

use crate::bvh::{BBox, BVHBuilder, FlatBVHNode, SplitMethod};
use crate::materials::MaterialRef;
use crate::types::ray::Ray;
use na::{Matrix4, Point3, Vector3};
use tri_mesh::Triangle;
use sphere::Sphere;
use quad_mesh::Quad;
use std::sync::Arc;

//pub type Primitive = Arc<dyn Hittable + Sync + Send>;

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

pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    fn mat(&self) -> Option<MaterialRef>;
    fn bbox(&self) -> BBox;
}

pub enum Primitive {
    Triangle(Triangle),
    Sphere(Sphere),
    Quad(Quad),
    Instance(Instance),
}

impl Hittable for Primitive {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        match self {
            Primitive::Triangle(tri) => tri.hit(ray, t_min, t_max),
            Primitive::Sphere(sph) => sph.hit(ray, t_min, t_max),
            Primitive::Quad(q) => q.hit(ray, t_min, t_max),
            Primitive::Instance(i) => i.hit(ray, t_min, t_max),
        }
    }

    fn mat(&self) -> Option<MaterialRef> {
        match self {
            Primitive::Triangle(tri) => tri.mat(),
            Primitive::Sphere(sph) => sph.mat(),
            Primitive::Quad(q) => q.mat(),
            Primitive::Instance(i) => i.mat(),
        }
    }

    fn bbox(&self) -> BBox {
        match self {
            Primitive::Triangle(tri) => tri.bbox(),
            Primitive::Sphere(sph) => sph.bbox(),
            Primitive::Quad(q) => q.bbox(),
            Primitive::Instance(i) => i.bbox(),
        }
    }
}

pub struct InnerPrimitiveBuffer<T: Hittable> {
    pub buffer: Vec<T>,
    pub bvh: Option<Box<[FlatBVHNode]>>,
    pub bbox: BBox
}

impl <T: Hittable> InnerPrimitiveBuffer<T> {
    pub fn new() -> InnerPrimitiveBuffer<T> {
        InnerPrimitiveBuffer {
            buffer: Vec::new(),
            bvh: None,
            bbox: BBox::empty(),
        }
    }

    pub fn push(&mut self, primitive: T) {
        self.bbox = self.bbox.merge(&primitive.bbox());
        self.buffer.push(primitive);
    }
}

pub struct PrimitiveBuffer {
    pub triangles: InnerPrimitiveBuffer<Triangle>,
    pub spheres: InnerPrimitiveBuffer<Sphere>,
    pub quads: InnerPrimitiveBuffer<Quad>,
    pub instances: InnerPrimitiveBuffer<Instance>,
}

impl PrimitiveBuffer {
    pub fn new() -> PrimitiveBuffer {
        PrimitiveBuffer {
            triangles: InnerPrimitiveBuffer::new(),
            spheres: InnerPrimitiveBuffer::new(),
            quads:  InnerPrimitiveBuffer::new(),
            instances: InnerPrimitiveBuffer::new(),
        }
    }

    pub fn add(&mut self, primitive: Primitive) {
        match primitive {
            Primitive::Triangle(tri) => self.triangles.push(tri),
            Primitive::Sphere(sph) => self.spheres.push(sph),
            Primitive::Quad(q) => self.quads.push(q),
            Primitive::Instance(i) => self.instances.push(i),
        }
    }

    pub fn build_bvh(&mut self) {
        if self.triangles.buffer.len() > 0 {
            BVHBuilder::build(SplitMethod::SAH, &mut self.triangles).unwrap();
        }

        if self.spheres.buffer.len() > 0 {
            BVHBuilder::build(SplitMethod::SAH, &mut self.spheres).unwrap();
        }

        if self.quads.buffer.len() > 0 {
            BVHBuilder::build(SplitMethod::SAH, &mut self.quads).unwrap();
        }

        if self.instances.buffer.len() > 0 {
            BVHBuilder::build(SplitMethod::SAH, &mut self.instances).unwrap();
        }
    }
}

impl Hittable for PrimitiveBuffer {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let tri_hit = self.triangles.hit(ray, t_min, t_max);
        let sph_hit = self.spheres.hit(ray, t_min, t_max);
        let q_hit = self.quads.hit(ray, t_min, t_max);
        let inst_hit = self.instances.hit(ray, t_min, t_max);

        fn compare(a: Option<HitRecord>, b: Option<HitRecord>) -> Option<HitRecord> {
            match (a, b) {
                (Some(a), Some(b)) => {
                    if a.t < b.t {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None
            }
        }

        compare(compare(compare(tri_hit, sph_hit), q_hit), inst_hit)
    }
    fn mat(&self) -> Option<MaterialRef> {
        None
    }
    fn bbox(&self) -> BBox {
        self.instances.bbox().merge(&self.triangles.bbox().merge(&self.spheres.bbox()).merge(&self.quads.bbox()))
    }
}

pub struct Instance {
    obj: PrimitiveBuffer,
    transform: Matrix4<f32>,
    inverse: Matrix4<f32>,
}

impl Instance {
    pub fn new(obj: PrimitiveBuffer, transform: na::Matrix4<f32>) -> Result<Self, &'static str> {
        let inverse = transform.try_inverse().ok_or("Matrix is not invertible")?;
        Ok(Instance {
            obj,
            transform,
            inverse,
        })
    }

    pub fn from_obj(obj: PrimitiveBuffer) -> Self {
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
