use na::{Matrix3, Vector3};

pub struct OrthonormalBasis {
    transform: Matrix3<f32>,
}

impl OrthonormalBasis {
    // y up
    pub fn new(n: &Vector3<f32>) -> Self {
        let v = n.normalize();
        let a = if v.x.abs() > 0.9 {
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let w = (&v).cross(&a).normalize();
        let u = (&v).cross(&w);

        let transform = Matrix3::from_columns(&[u, v, w]);

        Self { transform }
    }

    pub fn u(&self) -> Vector3<f32> {
        self.transform.column(0).into()
    }

    pub fn v(&self) -> Vector3<f32> {
        self.transform.column(1).into()
    }

    pub fn w(&self) -> Vector3<f32> {
        self.transform.column(2).into()
    }

    // Transform direction from world to local
    pub fn to_local(&self, v: &Vector3<f32>) -> Vector3<f32> {
        self.transform.transpose() * v
    }

    // Transform direction from local to world
    pub fn to_world(&self, v: &Vector3<f32>) -> Vector3<f32> {
        self.transform * v
    }
}
