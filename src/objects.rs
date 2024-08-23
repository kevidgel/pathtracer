pub mod quad_mesh;
pub mod sphere;
pub mod tri_mesh;

use crate::bvh::{BBox, BVHBuilder, FlatBVHNode, SplitMethod};
use crate::materials::MaterialRef;
use crate::types::ray::Ray;
use na::{Matrix4, Point3, Vector3};
use quad_mesh::Quad;
use sphere::Sphere;
use tri_mesh::Triangle;
use rand::Rng;

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
    fn hit(&self, ray: &mut Ray) -> Option<HitRecord>;
    fn mat(&self) -> Option<MaterialRef>;
    fn bbox(&self) -> BBox;
    fn pdf(&self, _ray: &Ray) -> f32 {
        0.0
    }
    fn sample(&self, _rng: &mut impl Rng, _origin: &Point3<f32>) -> Vector3<f32> {
        Vector3::zeros()
    }
}

pub struct InnerPrimitiveBuffer<T: Hittable> {
    pub buffer: Vec<T>,
    pub bvh: Option<Box<[FlatBVHNode]>>,
    pub bbox: BBox,
}

impl<T: Hittable> InnerPrimitiveBuffer<T> {
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

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct PrimitiveBuffer {
    pub triangles: InnerPrimitiveBuffer<Triangle>,
    pub spheres: InnerPrimitiveBuffer<Sphere>,
    pub quads: InnerPrimitiveBuffer<Quad>,
    pub instances: InnerPrimitiveBuffer<Instance>,
    pub bbox: BBox,
}

impl PrimitiveBuffer {
    pub fn new() -> PrimitiveBuffer {
        PrimitiveBuffer {
            triangles: InnerPrimitiveBuffer::new(),
            spheres: InnerPrimitiveBuffer::new(),
            quads: InnerPrimitiveBuffer::new(),
            instances: InnerPrimitiveBuffer::new(),
            bbox: BBox::empty(),
        }
    }

    pub fn add_triangle(&mut self, triangle: Triangle) {
        self.bbox = self.bbox.merge(&triangle.bbox());
        self.triangles.push(triangle);
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.bbox = self.bbox.merge(&sphere.bbox());
        self.spheres.push(sphere);
    }

    pub fn add_quad(&mut self, quad: Quad) {
        self.bbox = self.bbox.merge(&quad.bbox());
        self.quads.push(quad);
    }

    pub fn add_instance(&mut self, instance: Instance) {
        self.bbox = self.bbox.merge(&instance.bbox());
        self.instances.push(instance);
    }

    pub fn build_bvh(&mut self) {
        if self.triangles.len() > 0 {
            log::info!("Building BVH for {} triangles", self.triangles.len());
            BVHBuilder::build(SplitMethod::SAH, &mut self.triangles).unwrap();
        }

        if self.spheres.len() > 0 {
            log::info!("Building BVH for {} spheres", self.spheres.len());
            BVHBuilder::build(SplitMethod::SAH, &mut self.spheres).unwrap();
        }

        if self.quads.len() > 0 {
            log::info!("Building BVH for {} quads", self.quads.len());
            BVHBuilder::build(SplitMethod::SAH, &mut self.quads).unwrap();
        }

        if self.instances.len() > 0 {
            log::info!("Building BVH for {} instances", self.instances.len());
            BVHBuilder::build(SplitMethod::SAH, &mut self.instances).unwrap();
        }
    }

    pub fn hit(&self, ray: &mut Ray) -> Option<HitRecord> {
        let hit = self.triangles.hit(ray);
        let hit = self.spheres.hit(ray).or(hit);
        let hit = self.quads.hit(ray).or(hit);
        let hit = self.instances.hit(ray).or(hit);
        hit
    }

    pub fn bbox(&self) -> BBox {
        self.bbox
    }

    pub fn len(&self) -> usize {
        self.triangles.len() + self.spheres.len() + self.quads.len() + self.instances.len()
    }
}

pub struct LightBuffer {
    pub objects: PrimitiveBuffer,
}

impl LightBuffer {
    pub fn new() -> Self {
        Self {
            objects: PrimitiveBuffer::new(),
        }
    }

    pub fn add_triangle(&mut self, triangle: Triangle) {
        self.objects.add_triangle(triangle);
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.objects.add_sphere(sphere);
    }

    pub fn add_quad(&mut self, quad: Quad) {
        self.objects.add_quad(quad);
    }

    pub fn add_instance(&mut self, instance: Instance) {
        self.objects.add_instance(instance);
    }

    pub fn sample(&self, rng: &mut impl Rng, origin: &Point3<f32>) -> Vector3<f32> {
        let n = self.objects.len();
        let i = rng.gen_range(0..n);

        // Sample a primitive
        if i < self.objects.triangles.len() {
            self.objects.triangles.buffer[i].sample(rng, origin)
        } else if i < self.objects.triangles.len() + self.objects.spheres.len() {
            self.objects.spheres.buffer[i - self.objects.triangles.len()].sample(rng, origin)
        } else if i < self.objects.triangles.len() + self.objects.spheres.len() + self.objects.quads.len() {
            self.objects.quads.buffer[i - self.objects.triangles.len() - self.objects.spheres.len()].sample(rng, origin)
        } else {
            self.objects.instances.buffer[i - self.objects.triangles.len() - self.objects.spheres.len() - self.objects.quads.len()].sample(rng, origin)
        }
    }

    pub fn pdf(&self, ray: &Ray) -> f32 {
        let mut pdf = 0.0;
        for tri in &self.objects.triangles.buffer {
            pdf += tri.pdf(ray);
        }

        for sphere in &self.objects.spheres.buffer {
            pdf += sphere.pdf(ray);
        }

        for quad in &self.objects.quads.buffer {
            pdf += quad.pdf(ray);
        }

        for instance in &self.objects.instances.buffer {
            pdf += instance.pdf(ray);
        }

        pdf / (self.objects.len() as f32)
    }
}


#[repr(align(32))]
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
    fn hit(&self, ray: &mut Ray) -> Option<HitRecord> {
        let mut new_ray = Ray::new_bounded(
            self.inverse.transform_point(&ray.origin),
            self.inverse.transform_vector(&ray.direction),
            ray.t_min,
            ray.t_max,
        );
        match self.obj.hit(&mut new_ray) {
            Some(mut rec) => {
                rec.p = self.transform.transform_point(&rec.p);
                rec.normal = self.transform.transform_vector(&rec.normal);

                ray.t_max = rec.t;
                Some(rec)
            }
            None => None,
        }
    }

    fn mat(&self) -> Option<MaterialRef> {
        None
    }

    fn bbox(&self) -> BBox {
        self.obj.bbox().transform(self.transform)
    }
}
