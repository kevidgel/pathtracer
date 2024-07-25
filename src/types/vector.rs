use std::fmt;
use std::ops;

#[derive(Debug, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    pub fn zeros() -> Self {
        Self {
            x: 0_f32,
            y: 0_f32,
            z: 0_f32,
        }
    }

    pub fn ones() -> Self {
        Self {
            x: 1_f32,
            y: 1_f32,
            z: 1_f32,
        }
    }

    // Arithmetic
    pub fn dot(lhs: &Self, rhs: &Self) -> Self {
        Self {
            x: lhs.x * rhs.x,
            y: lhs.y * rhs.y,
            z: lhs.z * rhs.z,
        }
    }

    pub fn cross(lhs: &Self, rhs: &Self) -> Self {
        Self {
            x: lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.z * rhs.x - lhs.x * rhs.z,
            z: lhs.x * rhs.y - lhs.y * rhs.x,
        }
    }

    pub fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn unit(&self) -> Self {
        let norm: f32 = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }
}

/* Common Operations */

// Indexing
impl ops::Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("out of bounds"),
        }
    }
}

impl ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("out of bounds"),
        }
    }
}

// Arithmetic
impl ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3::new(self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2])
    }
}

impl ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3::new(self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2])
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3::new(self[0] * rhs, self[1] * rhs, self[2] * rhs)
    }
}

impl ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl ops::Mul for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Self) -> Self::Output {
        Vec3::new(self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2])
    }
}

impl ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Self::Output {
        Vec3::new(self[0] / rhs, self[1] / rhs, self[2] / rhs)
    }
}

impl ops::Div for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: Self) -> Self::Output {
        Vec3::new(self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2])
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3::new(-self[0], -self[1], -self[2])
    }
}

// Assignment arithmetic
impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
    }
}

impl ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
    }
}

impl ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
    }
}

impl ops::MulAssign for Vec3 {
    fn mul_assign(&mut self, rhs: Self) {
        self[0] *= rhs[0];
        self[1] *= rhs[1];
        self[2] *= rhs[2];
    }
}

impl ops::DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self[0] /= rhs;
        self[1] /= rhs;
        self[2] /= rhs;
    }
}

impl ops::DivAssign for Vec3 {
    fn div_assign(&mut self, rhs: Self) {
        self[0] /= rhs[0];
        self[1] /= rhs[1];
        self[2] /= rhs[2];
    }
}

// Printing
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self[0], self[1], self[2])
    }
}
