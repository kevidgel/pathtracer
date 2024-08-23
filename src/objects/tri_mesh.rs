// Will be painful...
// Use obj for now since its simple
// Also not really a "mesh"...

use crate::bvh::BBox;
use crate::materials::MaterialRef;
use crate::objects::Hittable;
use crate::types::ray::Ray;
use na::{ArrayStorage, Matrix, Point2, Point3, Vector3, U1, U3, U8};
use tobj;

use super::HitRecord;

type TriangleData = Matrix<f32, U8, U3, ArrayStorage<f32, 8, 3>>;
type VertexData = Matrix<f32, U8, U1, ArrayStorage<f32, 8, 1>>;

fn get_normal(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> Vector3<f32> {
    (v1 - v0).cross(&(v2 - v0)).normalize()
}

#[repr(align(32))]
#[derive(Clone)]
pub struct Triangle {
    data: TriangleData,
    material: Option<MaterialRef>,
}

impl Triangle {
    pub fn new(
        vertices: [Point3<f32>; 3],
        normals: Option<[Vector3<f32>; 3]>,
        texture_uv: Option<[Point2<f32>; 3]>,
        material: Option<MaterialRef>,
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

        let data: TriangleData = TriangleData::from_row_slice(&[
            vertices[0].x,
            vertices[1].x,
            vertices[2].x,
            vertices[0].y,
            vertices[1].y,
            vertices[2].y,
            vertices[0].z,
            vertices[1].z,
            vertices[2].z,
            normals[0].x,
            normals[1].x,
            normals[2].x,
            normals[0].y,
            normals[1].y,
            normals[2].y,
            normals[0].z,
            normals[1].z,
            normals[2].z,
            texture_uv[0].x,
            texture_uv[1].x,
            texture_uv[2].x,
            texture_uv[0].y,
            texture_uv[1].y,
            texture_uv[2].y,
        ]);

        Self { data, material }
    }

    pub fn new_from_vertex(
        a: &Vertex,
        b: &Vertex,
        c: &Vertex,
        material: Option<MaterialRef>,
    ) -> Self {
        let data: TriangleData = TriangleData::from_row_slice(&[
            a.position().x,
            b.position().x,
            c.position().x,
            a.position().y,
            b.position().y,
            c.position().y,
            a.position().z,
            b.position().z,
            c.position().z,
            a.normal().x,
            b.normal().x,
            c.normal().x,
            a.normal().y,
            b.normal().y,
            c.normal().y,
            a.normal().z,
            b.normal().z,
            c.normal().z,
            a.uv().x,
            b.uv().x,
            c.uv().x,
            a.uv().y,
            b.uv().y,
            c.uv().y,
        ]);

        Self { data, material }
    }

    pub fn position(&self, vertex: usize) -> Point3<f32> {
        Point3::new(
            self.data[(0, vertex)],
            self.data[(1, vertex)],
            self.data[(2, vertex)],
        )
    }

    pub fn normal(&self, vertex: usize) -> Vector3<f32> {
        Vector3::new(
            self.data[(3, vertex)],
            self.data[(4, vertex)],
            self.data[(5, vertex)],
        )
    }

    pub fn uv(&self, vertex: usize) -> Point2<f32> {
        Point2::new(self.data[(6, vertex)], self.data[(7, vertex)])
    }

    pub fn bary_interpolate(&self, bary_coords: &Vector3<f32>) -> Vertex {
        Vertex {
            data: &self.data * bary_coords,
        }
    }
}

pub struct Vertex {
    data: VertexData,
}

impl Vertex {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, uv: Point2<f32>) -> Self {
        let data: VertexData = VertexData::from_row_slice(&[
            position.x, position.y, position.z, normal.x, normal.y, normal.z, uv.x, uv.y,
        ]);

        Self { data }
    }

    pub fn position(&self) -> Point3<f32> {
        Point3::new(self.data[(0, 0)], self.data[(1, 0)], self.data[(2, 0)])
    }

    pub fn normal(&self) -> Vector3<f32> {
        Vector3::new(self.data[(3, 0)], self.data[(4, 0)], self.data[(5, 0)])
    }

    pub fn uv(&self) -> Point2<f32> {
        Point2::new(self.data[(6, 0)], self.data[(7, 0)])
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &mut Ray) -> Option<HitRecord> {
        // Moller-Trumbore algorithm
        let v0 = &self.position(0);
        let v1 = &self.position(1);
        let v2 = &self.position(2);

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let d = &ray.direction;

        let e1_x_d = &e1.cross(&d);

        let triple = (&e1_x_d).dot(&e2);
        if triple == 0.0 {
            return None;
        }

        let inv_triple = 1.0 / triple;
        let s = &ray.origin - v0;
        let e2_x_s = &e2.cross(&s);
        let u = (&e2_x_s).dot(&d) * inv_triple;
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let v = (&e1_x_d).dot(&s) * inv_triple;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let uvt = Vector3::new(u, v, (&e2_x_s).dot(&e1) * inv_triple);

        match ray.t_min <= uvt[2] && uvt[2] <= ray.t_max {
            true => {
                // Interpolate
                // TODO: Rewrite this using BLAS
                let t = uvt[2];
                let wuv = Vector3::new(1.0 - uvt[0] - uvt[1], uvt[0], uvt[1]);
                let interpolated = self.bary_interpolate(&wuv);
                let normal = interpolated.normal().normalize();
                let position = interpolated.position();
                let tex_coord = interpolated.uv();
                ray.t_max = t;
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

    fn mat(&self) -> Option<MaterialRef> {
        self.material.clone()
    }

    fn bbox(&self) -> BBox {
        let min = Point3::new(
            self.position(0)
                .x
                .min(self.position(1).x.min(self.position(2).x)),
            self.position(0)
                .y
                .min(self.position(1).y.min(self.position(2).y)),
            self.position(0)
                .z
                .min(self.position(1).z.min(self.position(2).z)),
        );
        let max = Point3::new(
            self.position(0)
                .x
                .max(self.position(1).x.max(self.position(2).x)),
            self.position(0)
                .y
                .max(self.position(1).y.max(self.position(2).y)),
            self.position(0)
                .z
                .max(self.position(1).z.max(self.position(2).z)),
        );
        BBox::new(min, max)
    }
}

pub struct TriMesh {
    positions: (Vec<Point3<f32>>, Vec<u32>),
    normals: Option<(Vec<Vector3<f32>>, Vec<u32>)>,
    uvs: Option<(Vec<Point2<f32>>, Vec<u32>)>,
}

impl TriMesh {
    pub fn new(
        positions: (Vec<Point3<f32>>, Vec<u32>),
        normals: Option<(Vec<Vector3<f32>>, Vec<u32>)>,
        uvs: Option<(Vec<Point2<f32>>, Vec<u32>)>,
    ) -> Self {
        Self {
            positions,
            normals,
            uvs,
        }
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
                log::error!(
                    "Failed to load materials: {}... Using empty materials instead...",
                    e
                );
                vec![]
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

    pub fn to_triangles_with_mat(&self, material: MaterialRef) -> Vec<Triangle> {
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
