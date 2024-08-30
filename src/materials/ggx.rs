// We use http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
use na::{Vector3, Point3};
use super::Material;
use crate::objects::HitRecord;
use crate::textures::{TextureRef, Solid};
use crate::types::pdf::{PDF, CosineWeightedHemispherePDF};
use crate::types::{color::{Color, ColorOps}, ray::Ray};
use rand::rngs::ThreadRng;
use std::sync::Arc;

pub struct GGX {
    texture: TextureRef, // Texture map 
    ior: f32, // Index of Refraction
    roughness: f32, // Roughness
    metallic: f32, // Metallic (TODO: Map)
    k_d: f32, // Diffuse 
    k_s: f32, // Specular
}

impl GGX {
    pub fn new(albedo: Color, ior: f32, roughness: f32, metallic: f32, k_d: f32, k_s: f32) -> Self {
        Self {
            texture: Arc::new(Solid::new(albedo)),
            ior,
            roughness,
            metallic,
            k_d,
            k_s
        }
    }

    fn bsdf_lambertian(&self, _ray_out: &Ray, _ray_in: &Ray, rec: &HitRecord) -> Color {
        self.texture.value(rec.u(), rec.v(), &rec.p()) / std::f32::consts::PI
    }

    fn bsdf_cook_torrance(&self, ray_out: &Ray, ray_in: &Ray, rec: &HitRecord) -> Color {
        let half_vector = (ray_out.direction + ray_in.direction).normalize();

        let cos_in = ray_in.direction.normalize().dot(&rec.normal()).clamp(0.00001, 1.0);
        let cos_out = (-ray_out.direction.normalize().dot(&rec.normal())).clamp(0.00001, 1.0);
        let cos_t = half_vector.dot(&ray_out.direction.normalize()).clamp(0.0, 1.0);

        let distribution = self.ggx_dist(rec.normal(), half_vector, self.roughness);
        let fresnel = self.fresnel_schlick(cos_t, self.ior, rec);
        let geometry = self.ggx_partial_geom(ray_out.direction, rec.normal(), half_vector, self.roughness)
            * self.ggx_partial_geom(ray_in.direction, rec.normal(), half_vector, self.roughness);

        let dfg = distribution * geometry * fresnel;


        let out = (dfg) / (4.0 * cos_in * cos_out);

        if out.x.is_nan() || out.y.is_nan() || out.z.is_nan() {
            log::error!("nan cook torrance! {} {}", cos_in, cos_out);
            return Color::zeros();
        }

        out
    }

    fn chi_ggx(&self, v: f32) -> f32 {
        if v > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn ggx_dist(&self, n: Vector3<f32>, h: Vector3<f32>, alpha: f32) -> f32 {
        let n_dot_h = n.dot(&h);
        let alpha_2 = alpha * alpha; 
        let n_dot_h_2 = n_dot_h * n_dot_h;
        let den = n_dot_h_2 * alpha_2 + (1.0 - n_dot_h_2);

        (self.chi_ggx(n_dot_h) * alpha_2) / (std::f32::consts::PI * den * den)
    } 

    fn ggx_partial_geom(&self, v: Vector3<f32>, n: Vector3<f32>, h: Vector3<f32>, alpha: f32) -> f32 {
        let v_dot_h = v.dot(&h).clamp(0.0, 1.0);
        let chi = self.chi_ggx(v_dot_h / v.dot(&n).clamp(0.0, 1.0));
        let v_dot_h_2 = v_dot_h * v_dot_h;
        let tan_2 = (1.0 - v_dot_h_2) / v_dot_h_2;

        (chi * 2.0) / (1.0 + (1.0 + alpha * alpha * tan_2).sqrt()) 
    }

    fn fresnel_schlick(&self, cos_t: f32, ior: f32, rec: &HitRecord) -> Vector3<f32> {
        let f_0 = ((1.0 - ior) / (1.0 + ior)).abs();
        let f_0 = f_0 * f_0;
        let f_0 = Vector3::new(f_0, f_0, f_0);
        // If metallic is 1, we just use the albedo
        let f_0 = f_0.lerp(&self.texture.value(rec.u(), rec.v(), &rec.p()), self.metallic);

        f_0 + (Vector3::new(1.0, 1.0, 1.0) - f_0) * (1.0 - cos_t).powf(5.0)
    }
}

impl Material for GGX {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn scatter(&self, rng: &mut ThreadRng, _ray_out: &Ray, rec: &HitRecord) -> Ray {
        let cosine_pdf = CosineWeightedHemispherePDF::new(&rec.normal());

        // Scatter
        let scatter_direction = &cosine_pdf.generate(rng);
        let scattered = Ray::new(rec.p(), scatter_direction.normalize());

        scattered
    }

    fn bsdf_evaluate(&self, ray_out: &Ray, ray_in: &Ray, rec: &HitRecord) -> Color {
        self.k_d * self.bsdf_lambertian(ray_out, ray_in, rec) +
            self.k_s * self.bsdf_cook_torrance(ray_out, ray_in, rec)
    }

    fn scattering_pdf(&self, _ray_out: &Ray, ray_in: &Ray, rec: &HitRecord) -> f32 {
        rec.normal().dot(&ray_in.direction.normalize()).max(0.0) / std::f32::consts::PI
    }
}
