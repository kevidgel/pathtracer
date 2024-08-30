use crate::camera::Camera;
use crate::objects::{LightBuffer, PrimitiveBuffer};

pub mod cornell;
pub mod lucy;
pub mod object;

pub trait Scene {
    fn build_scene() -> (PrimitiveBuffer, LightBuffer);
    fn build_camera() -> Camera;
}
