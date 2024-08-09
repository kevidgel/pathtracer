use crate::bvh::{BVHBuilder, FlatBVH, SplitMethod, BVH};
use crate::camera::Camera;
use crate::objects::HittableObjects;

pub mod cornell;
pub mod lucy;

pub trait Scene {
    fn build_scene() -> HittableObjects;
    fn build_scene_bvh() -> BVH {
        let objects = Self::build_scene();
        BVHBuilder::build_from_hittable_objects(SplitMethod::SAH, objects)
    }
    fn build_scene_flat_bvh() -> Result<FlatBVH, ()> {
        let objects = Self::build_scene();
        BVHBuilder::build_flattened_from_hittable_objects(SplitMethod::SAH, objects)
    }
    fn build_camera() -> Camera;
}
