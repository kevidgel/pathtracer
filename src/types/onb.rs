use na::{Matrix3, Vector3};

pub struct OrthonormalBasis {
    u: Vector3<f32>,
    v: Vector3<f32>,
    w: Vector3<f32>,
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

        Self { u, v, w }
    }

    pub fn u(&self) -> Vector3<f32> {
        self.u
    }

    pub fn v(&self) -> Vector3<f32> {
       self.v
    }

    pub fn w(&self) -> Vector3<f32> {
        self.w
    }

    pub fn to_world(&self, v: &Vector3<f32>) -> Vector3<f32> {
        (&self.u * v.x) + (&self.v * v.y) + (&self.w * v.z)
    }
}