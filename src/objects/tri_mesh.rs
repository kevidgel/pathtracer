// Will be painful...
// Use obj for now since its simple
// Also not really a "mesh"...

use crate::bvh::BBox;
use crate::materials::Material;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{Matrix4, Point2, Point3, Vector2, Vector3};
use rayon::vec;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tobj;

use super::HitRecord;

#[derive(Clone, Copy, Debug)]
pub struct TriVert {
    position: Point3<f32>,
    normal: Vector3<f32>,
    uv: Point2<f32>,
}

impl TriVert {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, uv: Point2<f32>) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }

    pub fn set_uv(&mut self, uv: Point2<f32>) {
        self.uv = uv;
    }

    pub fn set_normal(&mut self, normal: Vector3<f32>) {
        self.normal = normal;
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
        texture_uv: Option<[Point2<f32>; 3]>,
        material: Option<Arc<dyn Material + Sync + Send>>,
    ) -> Self {
        let normals = normals.unwrap_or([
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
            get_normal(&vertices[0], &vertices[1], &vertices[2]),
        ]);

        let texture_uv = texture_uv.unwrap_or([
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 0.0),
        ]);

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

    // pub fn transform_mut(&mut self, transform: &Matrix4<f32>) {
    //     for v in self.vertices.iter_mut() {
    //         v.position = transform * Point3::from_vec(v.position.coords);
    //     }
    // }
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
                // Interpolate
                let (u, v, w, t) = (uvt[0], uvt[1], 1.0 - uvt[0] - uvt[1], uvt[2]);
                let normal = (w * self.vertices[0].normal
                    + u * self.vertices[1].normal
                    + v * self.vertices[2].normal)
                    .normalize();
                let position = v0 + u * e1 + v * e2;
                let tex_coord = w * self.vertices[0].uv.coords
                    + u * self.vertices[1].uv.coords
                    + v * self.vertices[2].uv.coords;
                Some(HitRecord::new(
                    ray,
                    position,
                    normal,
                    t,
                    self.material.clone(),
                    tex_coord.x,
                    tex_coord.y,
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
                .position
                .x
                .min(self.vertices[1].position.x.min(self.vertices[2].position.x)),
            self.vertices[0]
                .position
                .y
                .min(self.vertices[1].position.y.min(self.vertices[2].position.y)),
            self.vertices[0]
                .position
                .z
                .min(self.vertices[1].position.z.min(self.vertices[2].position.z)),
        );
        let max = Point3::new(
            self.vertices[0]
                .position
                .x
                .max(self.vertices[1].position.x.max(self.vertices[2].position.x)),
            self.vertices[0]
                .position
                .y
                .max(self.vertices[1].position.y.max(self.vertices[2].position.y)),
            self.vertices[0]
                .position
                .z
                .max(self.vertices[1].position.z.max(self.vertices[2].position.z)),
        );
        BBox::new(min, max)
    }
}

pub struct TriMesh {
    positions: (Vec<Point3<f32>>, Vec<u32>),
    normals: Option<(Vec<Vector3<f32>>, Vec<u32>)>,
    uvs: Option<(Vec<Point2<f32>>, Vec<u32>)>,
    transform: Matrix4<f32>,
}

impl TriMesh {
    pub fn new(
        positions: (Vec<Point3<f32>>, Vec<u32>),
        normals: Option<(Vec<Vector3<f32>>, Vec<u32>)>,
        uvs: Option<(Vec<Point2<f32>>, Vec<u32>)>,
        transform: Option<Matrix4<f32>>,
    ) -> Self {
        Self {
            positions,
            normals,
            uvs,
            transform: transform.unwrap_or(Matrix4::identity()),
        }
    }

    pub fn transform_mut(&mut self, transform: &Matrix4<f32>) {
        self.transform = self.transform * transform;
    }

    pub fn load_as_vec(path: &str) -> Vec<Self> {
        log::debug!("Loading mesh: {}...", path);
        let (models, materials) = match tobj::load_obj(path, &tobj::LoadOptions::default()) {
            Ok(res) => res,
            Err(e) => {
                log::error!("Failed to load mesh: {}... Using empty mesh instead...", e);
                return vec![];
            }
        };

        let materials = match materials {
            Ok(mat) => mat,
            Err(e) => {
                log::error!("Failed to load mesh: {}... Using empty mesh instead...", e);
                return vec![];
            }
        };

        log::debug!(
            "Found {} models and {} materials",
            models.len(),
            materials.len()
        );

        let mut meshes: Vec<Self> = vec![];
        for (i, m) in models.iter().enumerate() {
            let mesh_obj = &m.mesh;
            log::debug!(
                "Model {} has {} vertices and {} triangles",
                i,
                mesh_obj.positions.len() / 3,
                mesh_obj.indices.len() / 3
            );
            log::debug!(
                "Model {} has {} normals and {} uvs",
                i,
                mesh_obj.normals.len() / 3,
                mesh_obj.texcoords.len() / 3
            );
            // TODO: add normals and uvs
            let mesh = TriMesh::new(
                (
                    mesh_obj
                        .positions
                        .chunks(3)
                        .map(|v| Point3::new(v[0], v[1], v[2]))
                        .collect(),
                    mesh_obj.indices.clone(),
                ),
                if mesh_obj.normals.len() > 0 {
                    Some((
                        mesh_obj
                            .normals
                            .chunks(3)
                            .map(|v| Vector3::new(v[0], v[1], v[2]))
                            .collect(),
                        mesh_obj.normal_indices.clone(),
                    ))
                } else {
                    None
                },
                if mesh_obj.texcoords.len() > 0 {
                    Some((
                        mesh_obj
                            .texcoords
                            .chunks(2)
                            .map(|v| Point2::new(v[0], v[1]))
                            .collect(),
                        mesh_obj.texcoord_indices.clone(),
                    ))
                } else {
                    None
                },
                None,
            );

            meshes.push(mesh);
        }

        meshes
    }

    pub fn to_triangles(&self) -> Vec<Triangle> {
        let range = self.positions.1.len();

        (0..range)
            .step_by(3)
            .map(|i| {
                Triangle::new(
                    [
                        self.positions.0[self.positions.1[i] as usize],
                        self.positions.0[self.positions.1[i + 1] as usize],
                        self.positions.0[self.positions.1[i + 2] as usize],
                    ],
                    match &self.normals {
                        Some(n) => Some([
                            n.0[n.1[i] as usize],
                            n.0[n.1[i + 1] as usize],
                            n.0[n.1[i + 2] as usize],
                        ]),
                        None => None,
                    },
                    match &self.uvs {
                        Some(uvs) => Some([
                            uvs.0[uvs.1[i] as usize],
                            uvs.0[uvs.1[i + 1] as usize],
                            uvs.0[uvs.1[i + 2] as usize],
                        ]),
                        None => None,
                    },
                    None,
                )
            })
            .collect()
    }

    pub fn to_triangles_with_mat(
        &self,
        material: Arc<dyn Material + Sync + Send>,
    ) -> Vec<Triangle> {
        let range = self.positions.1.len();

        (0..range)
            .step_by(3)
            .map(|i| {
                Triangle::new(
                    [
                        self.positions.0[self.positions.1[i] as usize],
                        self.positions.0[self.positions.1[i + 1] as usize],
                        self.positions.0[self.positions.1[i + 2] as usize],
                    ],
                    match &self.normals {
                        Some(n) => Some([
                            n.0[n.1[i] as usize],
                            n.0[n.1[i + 1] as usize],
                            n.0[n.1[i + 2] as usize],
                        ]),
                        None => None,
                    },
                    match &self.uvs {
                        Some(uvs) => Some([
                            uvs.0[uvs.1[i] as usize],
                            uvs.0[uvs.1[i + 1] as usize],
                            uvs.0[uvs.1[i + 2] as usize],
                        ]),
                        None => None,
                    },
                    Some(material.clone()),
                )
            })
            .collect()
    }
}
