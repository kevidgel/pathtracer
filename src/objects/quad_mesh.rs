use std::sync::Arc;

use crate::bvh::BBox;
use crate::materials::MaterialRef;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Point3, Vector3};

use super::{tri_mesh::Triangle, HitRecord, HittableObjects, Primitive};

#[derive(Clone)]
pub struct Quad {
    origin: Point3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    normal: Vector3<f32>,
    w: Vector3<f32>,
    d: f32,
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
        Self {
            origin: *origin,
            u: *u,
            v: *v,
            normal,
            w,
            d,
            mat,
            bbox,
        }
    }

    pub fn new_box(a: &Point3<f32>, b: &Point3<f32>, mat: Option<MaterialRef>) -> HittableObjects {
        let mut sides = HittableObjects::new();

        let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        let dx = Vector3::new(max.x - min.x, 0.0, 0.0);
        let dy = Vector3::new(0.0, max.y - min.y, 0.0);
        let dz = Vector3::new(0.0, 0.0, max.z - min.z);

        sides.add(Arc::new(Quad::new(
            &Point3::new(min.x, min.y, max.z),
            &dx,
            &dy,
            mat.clone(),
        )) as Primitive);
        sides.add(Arc::new(Quad::new(
            &Point3::new(max.x, min.y, max.z),
            &-dz,
            &dy,
            mat.clone(),
        )) as Primitive);
        sides.add(Arc::new(Quad::new(
            &Point3::new(max.x, min.y, min.z),
            &-dx,
            &dy,
            mat.clone(),
        )) as Primitive);
        sides.add(Arc::new(Quad::new(
            &Point3::new(min.x, min.y, min.z),
            &dz,
            &dy,
            mat.clone(),
        )) as Primitive);
        sides.add(Arc::new(Quad::new(
            &Point3::new(min.x, max.y, max.z),
            &dx,
            &-dz,
            mat.clone(),
        )) as Primitive);
        sides.add(Arc::new(Quad::new(
            &Point3::new(min.x, min.y, min.z),
            &dx,
            &dz,
            mat.clone(),
        )) as Primitive);

        sides
    }

    pub fn to_triangles(&self) -> Vec<Primitive> {
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
        vec![Arc::new(t1), Arc::new(t2)]
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let denom = self.normal.dot(&ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }

        let t = (self.d - self.normal.dot(&ray.origin.coords)) / denom;
        if t < t_min || t > t_max {
            return None;
        }

        let intersection = ray.at(t);
        let p = intersection - self.origin;

        let alpha = self.w.dot(&(p.cross(&self.v)));
        let beta = self.w.dot(&(self.u.cross(&p)));

        if !(0.0..=1.0).contains(&alpha) || !(0.0..=1.0).contains(&beta) {
            return None;
        }

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
}
