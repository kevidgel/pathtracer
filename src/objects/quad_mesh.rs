use crate::bvh::BBox;
use crate::materials::MaterialRef;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Point3, Vector3};
use rand::Rng;

use super::{tri_mesh::Triangle, HitRecord, PrimitiveBuffer};

#[repr(align(32))]
#[derive(Clone)]
pub struct Quad {
    origin: Point3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    normal: Vector3<f32>,
    w: Vector3<f32>,
    d: f32,
    area: f32,
    mat: Option<MaterialRef>,
    bbox: BBox,
}

impl Quad {
    pub fn new(
        origin: &Point3<f32>,
        u: &Vector3<f32>,
        v: &Vector3<f32>,
        mat: Option<MaterialRef>,
    ) -> Self {
        let bbox_diag1 = BBox::new(*origin, origin + u + v);

        let bbox_diag2 = BBox::new(origin + u, origin + v);
        let bbox = bbox_diag1.merge(&bbox_diag2);

        let n = u.cross(&v);
        let normal = n.normalize();
        let d = normal.dot(&origin.coords);
        let w = n / (n.dot(&n));
        let area = n.norm();
        Self {
            origin: *origin,
            u: *u,
            v: *v,
            normal,
            w,
            d,
            area,
            mat,
            bbox,
        }
    }

    pub fn new_box(a: &Point3<f32>, b: &Point3<f32>, mat: Option<MaterialRef>) -> PrimitiveBuffer {
        let mut sides = PrimitiveBuffer::new();

        let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        let dx = Vector3::new(max.x - min.x, 0.0, 0.0);
        let dy = Vector3::new(0.0, max.y - min.y, 0.0);
        let dz = Vector3::new(0.0, 0.0, max.z - min.z);

        sides.add_quad(Quad::new(
            &Point3::new(min.x, min.y, max.z),
            &dx,
            &dy,
            mat.clone(),
        ));
        sides.add_quad(Quad::new(
            &Point3::new(max.x, min.y, max.z),
            &-dz,
            &dy,
            mat.clone(),
        ));
        sides.add_quad(Quad::new(
            &Point3::new(max.x, min.y, min.z),
            &-dx,
            &dy,
            mat.clone(),
        ));
        sides.add_quad(Quad::new(
            &Point3::new(min.x, min.y, min.z),
            &dz,
            &dy,
            mat.clone(),
        ));
        sides.add_quad(Quad::new(
            &Point3::new(min.x, max.y, max.z),
            &dx,
            &-dz,
            mat.clone(),
        ));
        sides.add_quad(Quad::new(
            &Point3::new(min.x, min.y, min.z),
            &dx,
            &dz,
            mat.clone(),
        ));

        sides
    }

    pub fn to_triangles(&self) -> Vec<Triangle> {
        let t1 = Triangle::new(
            [self.origin, self.origin + self.u, self.origin + self.v],
            None,
            None,
            self.mat.clone(),
        );
        let t2 = Triangle::new(
            [
                self.origin + self.u,
                self.origin + self.u + self.v,
                self.origin + self.v,
            ],
            None,
            None,
            self.mat.clone(),
        );
        vec![t1, t2]
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &mut Ray) -> Option<HitRecord> {
        let denom = self.normal.dot(&ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }

        let t = (self.d - self.normal.dot(&ray.origin.coords)) / denom;
        if t < ray.t_min || t > ray.t_max {
            return None;
        }

        let intersection = ray.at(t);
        let p = intersection - self.origin;

        let alpha = self.w.dot(&(p.cross(&self.v)));
        let beta = self.w.dot(&(self.u.cross(&p)));

        if !(0.0..=1.0).contains(&alpha) || !(0.0..=1.0).contains(&beta) {
            return None;
        }

        ray.t_max = t;

        Some(HitRecord::new(
            ray,
            intersection,
            self.normal,
            t,
            self.mat.clone(),
            alpha,
            beta,
        ))
    }
    fn bbox(&self) -> BBox {
        self.bbox
    }
    fn mat(&self) -> Option<MaterialRef> {
        self.mat.clone()
    }
    fn pdf(&self, ray: &Ray) -> f32 {
        match self.hit(&mut ray.clone()) {
            Some(rec) => {
                let distance_squared = rec.t() * rec.t() * ray.direction.norm_squared();
                let cosine = ray.direction.dot(&rec.normal()).abs() / ray.direction.norm();

                distance_squared / (cosine * self.area)
            }
            None => 0.0,
        }
    }
    fn sample(&self, rng: &mut impl Rng, origin: &Point3<f32>) -> Vector3<f32> {
        let p = self.origin + self.u * rng.gen_range(0.0..1.0) + self.v * rng.gen_range(0.0..1.0);
        p - origin
    }
}
