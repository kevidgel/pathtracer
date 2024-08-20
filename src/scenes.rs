use crate::camera::Camera;
use crate::objects::PrimitiveBuffer;

pub mod cornell;
pub mod lucy;

pub trait Scene<'a> {
    fn build_scene() -> PrimitiveBuffer;
    fn build_camera() -> Camera;
}
