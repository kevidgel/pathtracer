// Disney "principled" BRDF. Not necessarily physically based.
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Every parameter is clamped to (0.0, 1.0)

use na::Vector3;
use super::{reflect, Material};
use crate::{objects::HitRecord, textures::{Solid, TextureRef}, types::color::{Color, ColorOps}}; 
use rand::Rng;
use std::sync::Arc;

const INV_PI: f32 = (1.0) / std::f32::consts::PI;

fn schlick_weight(h_dot_l: f32) -> f32 {
    (1.0 - h_dot_l).powf(5.0)
}

fn schlick(ior: f32, h_dot_l: f32) -> f32 {
    let r_0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));

    r_0 + (1.0 - r_0) * schlick_weight(h_dot_l)
}

pub struct Disney {
    base_color: TextureRef, // Surface color
    subsurface: f32, // Controls diffuse shape using a subsurface approximation
    metallic: f32, // 0 = dielectric, 1 = metallic. Linear blend between two different models
    specular: f32, // Incident specular amount
    specular_tint: f32, // Tints incident specular towards base color
    roughness: f32, // Surface roughness
    anisotropic: f32, // 0 = isotropic, 1 = maximally anisotropic. Controls aspect ratio of specular highlight
    sheen: f32, // Additional grazing component, intended for cloth
    sheen_tint: f32, // Amount of tint sheen towards base color
    clearcoat: f32, // A second, special-purpose specular lobe
    clearcoat_gloss: f32, // 0 = "satin", 1 = "gloss". Controls clearcoat glossiness.
}

impl Disney {
}

impl Material for Disney {
}

pub struct DisneySheen {
    base_color: TextureRef,
    sheen: f32, // TODO: texturize
    sheen_tint: f32, // TODO: texturize
}

impl DisneySheen {
    pub fn new(base_color: Color, sheen: f32, sheen_tint: f32) -> Self {
        Self {
            base_color: Arc::new(Solid::new(base_color)),
            sheen,
            sheen_tint
        }
    }

    fn calculate_tint(&self, base_color: &Vector3<f32>) -> Vector3<f32> {
        let luminance = Vector3::new(0.3, 0.6, 1.0).dot(&base_color).clamp(0.0, 1.0);
        if luminance > 0.0 { (1.0 / luminance) * base_color } else { Vector3::new(1.0, 1.0, 1.0) }
    }

    fn evaluate_sheen(&self, _w_o: &Vector3<f32>, h: &Vector3<f32>, w_i: &Vector3<f32>, rec: &HitRecord) -> Vector3<f32> {
        if self.sheen <= 0.0 {
            return Vector3::zeros();
        }

        // Acquire base_color
        let base_color = self.base_color.value(rec.u(), rec.v(), &rec.p());

        let h_dot_w_i = h.dot(&w_i);
        let tint = self.calculate_tint(&base_color);

        self.sheen * Vector3::new(1.0, 1.0, 1.0).lerp(&tint, self.sheen_tint) * schlick_weight(h_dot_w_i) 
    }
}

impl Material for DisneySheen {
    fn is_specular(&self) -> bool {
       false 
    }

    fn is_emissive(&self) -> bool {
        false
    }

    fn scatter(&self, rng: &mut rand::prelude::ThreadRng, _w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        // Cosine weighted sampling
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        Vector3::new(x, y, z)
    }


    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize(); 
        self.evaluate_sheen(&w_out, &h, &w_in, &rec)
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        &w_in.normalize().y * INV_PI
    }
}

pub struct DisneyClearcoat {
    clearcoat: f32,
    clearcoat_gloss: f32, // TODO: Texturize
    a2: f32,
    a2_invlog2: f32
}

impl DisneyClearcoat {
    pub fn new(clearcoat: f32, clearcoat_gloss: f32) -> Self {
        let a = (1.0 - clearcoat_gloss) * 0.1 + (clearcoat_gloss * 0.001);
        let a2 = a * a;

        Self {
            clearcoat,
            clearcoat_gloss,
            a2,
            a2_invlog2: 1.0 / a2.log2()
        }
    }

    fn gtr1(&self, cos_d: f32, a2: f32) -> f32 {
        if a2 >= 1.0 {
            return INV_PI;
        }

        INV_PI * (a2 - 1.0) * self.a2_invlog2 / ((1.0 + (a2 - 1.0) * cos_d * cos_d))
    }

    fn separable_smith_ggx_g1(&self, cos_w: f32, a2: f32) -> f32 {
        2.0 / (1.0 + (a2 + (1.0 - a2) *  cos_w * cos_w).sqrt())
    }

    fn separable_g(&self, w: &Vector3<f32>) -> f32 {
        let x = 0.25 * w.x;
        let y = w.y;
        let z = 0.25 * w.z;

        2.0 / (1.0 + (((x * x) + (z * z)) / (y * y))).sqrt() 
    }

    fn evaluate_clearcoat(&self, w_o: &Vector3<f32>, h: &Vector3<f32>, w_i: &Vector3<f32>, _rec: &HitRecord) -> Color {
        // Return if not used
        if self.clearcoat <= 0.0 {
            return Color::zeros();
        }

        // Clamp values just in case
        let cos_o = w_o.y.abs().min(1.0);
        let cos_i = w_i.y.abs().min(1.0);
        let cos_d = h.dot(&w_i).abs().min(1.0);

        // Calculate alpha_g
        let a2 = self.a2;

        // Calculate terms in final BRDF
        let d = self.gtr1(cos_d, a2);
        let f = schlick(1.5, cos_d);
        let g_i = self.separable_g(w_i);
        let g_o = self.separable_g(w_o);
        let g = g_i * g_o;

        let value = (0.25 * self.clearcoat * d * f * g) / (cos_o * cos_i);

        Color::gray(value)
    }
}

impl Material for DisneyClearcoat {
    fn is_specular(&self) -> bool {
        false
    }

    fn is_emissive(&self) -> bool {
        false
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize(); 
        self.evaluate_clearcoat(&w_out, &h, &w_in, &rec) 
    }

    fn scatter(&self, rng: &mut rand::prelude::ThreadRng, w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        let a2 = self.a2;

        let u_0 = rng.gen_range(0.0..1.0);
        let u_1 = rng.gen_range(0.0..1.0);

        let cos_elevation2 = (1.0 - a2.powf(1.0 - u_0)) / (1.0 - a2);
        let sin_elevation = (1.0 - cos_elevation2).sqrt();
        let cos_elevation = cos_elevation2.sqrt();
        let azimuth = 2.0 * std::f32::consts::PI * u_1;
        let sin_azimuth = azimuth.sin();
        let cos_azimuth = azimuth.cos();

        let x = sin_elevation * cos_azimuth;
        let y = cos_elevation;
        let z = sin_elevation * sin_azimuth;

        let micro_normal = Vector3::new(x, y, z);

        reflect(&w_out, &micro_normal)
    }

    fn scattering_pdf(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        let a2 = self.a2;

        let h = (w_out + w_in).normalize();
        let cos_d = h.dot(&w_in).abs().min(1.0);
        let cos_h = h.y.abs().min(1.0);
        let d = self.gtr1(cos_d, a2);

        (d * cos_h) / (4.0 * cos_d)
    }
}

pub struct DisneyMetal {
    base_color: TextureRef,
    roughness: f32,
    anisotropic: f32,
    a_x: f32,
    a_z: f32,
    inv_a_x: f32,
    inv_a_z: f32,
}

impl DisneyMetal {
    pub fn new(base_color: Color, roughness: f32, anisotropic: f32) -> Self {
        const A_MIN: f32 = 0.0001;
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let inv_aspect = 1.0 / aspect;
        
        let a_x = (roughness * roughness * inv_aspect).max(A_MIN);
        let a_z = (roughness * roughness * aspect).max(A_MIN);

        Self {
            base_color: Arc::new(Solid::new(base_color)),
            roughness,
            anisotropic,
            a_x,
            a_z,
            inv_a_x: 1.0 / a_x,
            inv_a_z: 1.0 / a_z,
        }
    }
    
    fn ggx_distribution(&self, h: &Vector3<f32>) -> f32 {
        let k = (h.x * h.x * self.inv_a_x * self.inv_a_x) + (h.z * h.z * self.inv_a_z * self.inv_a_z) + h.y * h.y; 
        INV_PI * self.inv_a_x * self.inv_a_z * (1.0 / (k * k))
    }

    fn separable_g(&self, w: &Vector3<f32>) -> f32 {
        let x = self.a_x * w.x;
        let y = w.y;
        let z = self.a_z * w.z;

        2.0 / (1.0 + (1.0 + ((x * x) + (z * z)) / (y * y)).sqrt()) 
    }

    fn evaluate_metal(&self, w_o: &Vector3<f32>, h: &Vector3<f32>, w_i: &Vector3<f32>, rec: &HitRecord) -> Color {
        // Common values
        let cos_o = w_o.y.abs().min(1.0);
        let cos_i = w_i.y.abs().min(1.0);
        let cos_d = h.dot(&w_i).abs().min(1.0);
        let base_color = self.base_color.value(rec.u(), rec.v(), &rec.p()); 

        // Compute Cook-Torrance terms
        let f = base_color.lerp(&Color::new(1.0, 1.0, 1.0), schlick_weight(cos_d));
        let d = self.ggx_distribution(&h);
        let g = self.separable_g(&w_o) * self.separable_g(&w_i);

        (0.25 * f * d * g) / (cos_o * cos_i) 
    }
}

impl Material for DisneyMetal {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize(); 
        self.evaluate_metal(&w_out, &h, &w_in, &rec)
    }

    fn scatter(&self, rng: &mut rand::prelude::ThreadRng, w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        // VNDF from Heitz
        // Generate random samples
        let u1: f32 = rng.gen_range(0.0..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);

        // Transform to hemisphere configuration
        let w_hemi = Vector3::new(self.a_x * w_out.x, w_out.y, self.a_z * w_out.z).normalize();

        // Orthonormal basis
        let basis_1 = if w_hemi.y < 0.9999 { Vector3::new(0.0, 1.0, 0.0).cross(&w_hemi).normalize() } else { Vector3::new(1.0, 0.0, 0.0) };
        let basis_2 = w_hemi.cross(&basis_1);

        // Parameterize projected area
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;
        let t1 = r * phi.cos();
        let t2 = r * phi.sin();
        let s = 0.5 * (1.0 + w_hemi.y);
        let t2 = (1.0 - s) * (1.0 - t1 * t1).sqrt() + (s * t2);

        // Project onto hemisphere
        let n_h = (t1 * basis_1) + (t2 * basis_2) + (1.0 - (t1 * t1) - (t2 * t2)).max(0.0).sqrt() * w_hemi;

        // Normalize into original configuration
        let n_e = Vector3::new(self.a_x * n_h.x, n_h.y.max(0.0), self.a_z * n_h.z).normalize(); 

        reflect(&w_out, &n_e)
    }
    
    fn scattering_pdf(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        let h = (w_out + w_in).normalize(); 
        let d = self.ggx_distribution(&h); 
        let g_o = self.separable_g(&w_out); 
        let cos_o = w_out.y.abs().min(1.0); 

        (0.25 * d * g_o) / cos_o
    }
}

pub struct DisneyDiffuse {
    base_color: TextureRef,
    roughness: f32, // TODO: Texturize
    subsurface: f32, // TODO: Texturize
}

impl DisneyDiffuse {
    pub fn new(base_color: Color, roughness: f32, subsurface: f32) -> Self {
        Self {
            base_color: Arc::new(Solid::new(base_color)),
            roughness,
            subsurface,
        }
    }

    fn f_d(&self, fresnel_w: f32, f_d_90: f32) -> f32 { 
        1.0 + (f_d_90 - 1.0) * fresnel_w
    }

    fn f_ss(&self, fresnel_w: f32, f_ss_90: f32) -> f32 {
        1.0 + (f_ss_90 - 1.0) * fresnel_w
    }

    fn evaluate_diffuse(&self, w_o: &Vector3<f32>, h: &Vector3<f32>, w_i: &Vector3<f32>, rec: &HitRecord) -> Color {
        // Compute common values
        let roughness = self.roughness;

        // Clamp values just in case
        let cos_o = w_o.y.abs().min(1.0);
        let cos_i = w_i.y.abs().min(1.0);
        let cos_d = h.dot(&w_i).abs().min(1.0);
        
        let fresnel_i = schlick_weight(cos_i);
        let fresnel_o = schlick_weight(cos_o);

        // Compute base_diffuse contribution
        let f_base = if self.subsurface < 1.0 {
            let f_d_90 = 0.5 + (2.0 * cos_d * cos_d * roughness);
            self.f_d(fresnel_i, f_d_90) * self.f_d(fresnel_o, f_d_90)
        } else {
            0.0
        };

        // Compute subsurface contribution
        let f_subsurf = if self.subsurface > 0.0 {
            let f_ss_90 = cos_d * cos_d * roughness;
            let f_ss = self.f_ss(fresnel_o, f_ss_90) * self.f_ss(fresnel_i, f_ss_90);
            1.25 * (f_ss * ((1.0 / (cos_i + cos_o)) - 0.5) + 0.5)
        } else {
            0.0
        };

        // Weight the contribution according to subsurface
        let f_diffuse = ((1.0 - self.subsurface) * f_base) + (self.subsurface * f_subsurf);

        // Get albedo
        let albedo = self.base_color.value(rec.u(), rec.v(), &rec.p());

        // BRDF is just Lambertian scaled by Disney's diffuse term
        f_diffuse * albedo * INV_PI
    }


}

impl Material for DisneyDiffuse {
    fn is_specular(&self) -> bool {
       false 
    }

    fn is_emissive(&self) -> bool {
        false
    }

    fn scatter(&self, rng: &mut rand::prelude::ThreadRng, _w_out: &Vector3<f32>, _rec: &HitRecord) -> Vector3<f32> {
        // Cosine weighted sampling
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        Vector3::new(x, y, z)
    }


    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize(); 
        self.evaluate_diffuse(&w_out, &h, &w_in, &rec)
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        &w_in.normalize().y * INV_PI 
    }
}
