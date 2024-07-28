// Will be painful...
// Use obj for now since its simple
// Also not really a "mesh"...

use crate::bvh::BBox;
use crate::materials::Material;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Point, Point3, Vector3};
use rayon::vec;
use tobj;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use super::HitRecord;

#[derive(Clone, Copy, Debug)]
pub struct TriVert {
    position: Point3<f32>,
    normal: Vector3<f32>,
    uv: (f32, f32),
}

impl TriVert {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, uv: (f32, f32)) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }
}

fn get_normal(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> Vector3<f32> {
    (v1 - v0).cross(&(v2 - v0)).normalize()
}

pub struct Triangle {
    vertices: [TriVert; 3],
    material: Option<Arc<dyn Material + Sync + Send>>,
}

impl Triangle {
    pub fn new(
        vertices: [Point3<f32>; 3],
        normals: Option<[Vector3<f32>; 3]>,
        texture_uv: Option<[(f32, f32); 3]>,
        material: Option<Arc<dyn Material + Sync + Send>>,
    ) -> Self {
        let normals = normals.unwrap_or([
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
        ]);

        let texture_uv = texture_uv.unwrap_or([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);

        Self {
            vertices: [
                TriVert::new(vertices[0], normals[0], texture_uv[0]),
                TriVert::new(vertices[1], normals[1], texture_uv[1]),
                TriVert::new(vertices[2], normals[2], texture_uv[2]),
            ],
            material,
        }
    }

    pub fn vertices(&self) -> [TriVert; 3] {
        self.vertices
    }

    pub fn vertex(&self, index: usize) -> TriVert {
        self.vertices[index]
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Moller-Trumbore algorithm
        let v0 = self.vertices[0].position;
        let v1 = self.vertices[1].position;
        let v2 = self.vertices[2].position;

        let s = ray.origin - v0;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let d = ray.direction;

        let e1_x_d = e1.cross(&d);
        let s_x_e2 = s.cross(&e2);

        let triple = e1_x_d.dot(&e2);
        let uvt = Vector3::new((-s_x_e2).dot(&d), e1_x_d.dot(&s), (-s_x_e2).dot(&e1)) / triple;

        match (triple != 0.0)
            && (uvt[0] >= 0.0)
            && (uvt[1] >= 0.0)
            && (uvt[0] + uvt[1] <= 1.0)
            && t_min <= uvt[2]
            && uvt[2] <= t_max
        {
            true => {
                // Interpolated vertex normals
                let normal = ((1.0_f32 - uvt[0] - uvt[1]) * self.vertices[0].normal
                    + uvt[0] * self.vertices[1].normal
                    + uvt[1] * self.vertices[2].normal)
                    .normalize();
                // Barycentric
                let position = v0 + uvt[0] * e1 + uvt[1] * e2;
                Some(HitRecord::new(
                    ray,
                    position,
                    normal,
                    uvt[2],
                    self.material.clone(),
                    uvt[0],
                    uvt[1],
                ))
            }
            false => None,
        }
    }

    fn mat(&self) -> Option<Arc<dyn Material + Sync + Send>> {
        self.material.clone()
    }

    fn bbox(&self) -> BBox {
        let min = Point3::new(
            self.vertices[0]
                .position.x
                .min(self.vertices[1].position.x.min(self.vertices[2].position.x)),
            self.vertices[0]
                .position.y
                .min(self.vertices[1].position.y.min(self.vertices[2].position.y)),
            self.vertices[0]
                .position.z
                .min(self.vertices[1].position.z.min(self.vertices[2].position.z)),
        );
        let max = Point3::new(
            self.vertices[0]
                .position.x
                .max(self.vertices[1].position.x.max(self.vertices[2].position.x)),
            self.vertices[0]
                .position.y
                .max(self.vertices[1].position.y.max(self.vertices[2].position.y)),
            self.vertices[0]
                .position.z
                .max(self.vertices[1].position.z.max(self.vertices[2].position.z)),
        );
        BBox::new(min, max)
    }
}

pub struct TriMesh {
    vertices: Vec<TriVert>,
    indices: Vec<u32>,
}

impl TriMesh {
    pub fn new(vertices: Vec<TriVert>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn load(path: &str) -> Self {
        let (models, materials) = match tobj::load_obj(path, &tobj::LoadOptions::default()) {
            Ok((v, i)) => (v, i),
            Err(e) => {
                log::error!("Failed to load mesh: {}... Using empty mesh instead...", e);
                return Self::new(vec![], vec![]);
            }
        };

        Self::new(vec![], vec![])
    }
}
