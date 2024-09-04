// Disney "principled" BRDF. Not necessarily physically based.
// This isn't an exact implementation, there may be bugs
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

// Implementation notes:
// We use y-up (0, 1, 0)
// w_o (w_out) indicates the direction TO THE VIEWER
// w_i (w_in) indicates the direction TO THE LIGHT SOURCE
// All directions point out from the origin, and are transformed to the surface local (defined by the normal)
// w_o will always have y > 0.0. We indicate whether w_o is outside (true) or inside (false) the medium with rec.front_face()
// It is up to the user to clamp Disney BSDF parameters to 0.0, 1.0
// We only contribute BRDFs when w_i is reflected (w_i.y > 0) in the final material shader

use super::*;
use crate::{
    objects::HitRecord,
    textures::{Solid, TextureRef},
    types::color::{Color, ColorOps},
};
use na::Vector3;
use rand::Rng;
use std::sync::Arc;

const INV_PI: f32 = (1.0) / std::f32::consts::PI;

#[inline(always)]
pub fn r_0(ior: f32) -> f32 {
    let r_0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));

    r_0
}

#[inline(always)]
pub fn schlick_weight(h_dot_l: f32) -> f32 {
    (1.0 - h_dot_l).powf(5.0)
}

#[inline(always)]
pub fn schlick(ior: f32, h_dot_l: f32) -> f32 {
    let r_0 = r_0(ior);

    r_0 + (1.0 - r_0) * schlick_weight(h_dot_l)
}

#[inline(always)]
pub fn fresnel_o(w_o: &Vector3<f32>, n: &Vector3<f32>, eta: f32) -> f32 {
    // w_o will always be above surface (incident / transmission)
    let cos_o = w_o.dot(&n);

    let sin2_o = (1.0 - cos_o * cos_o).max(0.0);
    let sin2_i = sin2_o * eta * eta;

    if sin2_i >= 1.0 {
        1.0
    } else {
        let cos_o = cos_o.abs();
        let cos_i = (1.0 - sin2_i).max(0.0).sqrt();
        let eta = 1.0 / eta;
        let r_para = (cos_o - eta * cos_i) / (cos_o + eta * cos_i);
        let r_perp = (eta * cos_o - cos_i) / (eta * cos_o + eta * cos_i);

        0.5 * (r_para * r_para + r_perp * r_perp)
    }
}

#[inline(always)]
pub fn calculate_tint(base_color: &Vector3<f32>) -> Vector3<f32> {
    let luminance = Vector3::new(0.3, 0.6, 1.0).dot(&base_color).clamp(0.0, 1.0);
    if luminance > 0.0 {
        (1.0 / luminance) * base_color
    } else {
        Vector3::new(1.0, 1.0, 1.0)
    }
}

//============================================================================//

pub struct Disney {
    params: DisneyParameters,
    disney_diffuse: DisneyDiffuse,
    disney_metal: DisneyMetal,
    disney_glass: DisneyGlass,
    disney_clearcoat: DisneyClearcoat,
    disney_sheen: DisneySheen,
}

pub struct DisneyParameters {
    base_color: TextureRef,     // Surface color
    subsurface: f32,            // Controls diffuse shape using a subsurface approximation
    metallic: f32, // 0 = dielectric, 1 = metallic. Linear blend between two different models
    specular_transmission: f32, // Transmission amount
    specular: f32, // Incident specular amount
    specular_tint: f32, // Tints incident specular towards base color
    roughness: f32, // Surface roughness
    anisotropic: f32, // 0 = isotropic, 1 = maximally anisotropic. Controls aspect ratio of specular highlight
    sheen: f32,       // Additional grazing component, intended for cloth
    sheen_tint: f32,  // Amount of tint sheen towards base color
    clearcoat: f32,   // A second, special-purpose specular lobe
    clearcoat_gloss: f32, // 0 = "satin", 1 = "gloss". Controls clearcoat glossiness.
    ior: f32, // Index of refraction for the material, relative to air (1.0). Note: object must be a watertight mesh.
}

impl Disney {
    pub fn new(
        base_color: TextureRef,     // Surface color
        subsurface: f32,            // Controls diffuse shape using a subsurface approximation
        metallic: f32, // 0 = dielectric, 1 = metallic. Linear blend between two different models
        specular_transmission: f32, // Transmission amount
        specular: f32, // Incident specular amount
        specular_tint: f32, // Tints incident specular towards base color
        roughness: f32, // Surface roughness
        anisotropic: f32, // 0 = isotropic, 1 = maximally anisotropic. Controls aspect ratio of specular highlight
        sheen: f32,       // Additional grazing component, intended for cloth
        sheen_tint: f32,  // Amount of tint sheen towards base color
        clearcoat: f32,   // A second, special-purpose specular lobe
        clearcoat_gloss: f32, // 0 = "satin", 1 = "gloss". Controls clearcoat glossiness.
        ior: f32, // Index of refraction for the material, relative to air (1.0). Note: object must be a watertight mesh.
    ) -> Self {
        let params = DisneyParameters {
            base_color: base_color.clone(),
            subsurface,
            metallic,
            specular_transmission: specular_transmission.max(0.0001),
            specular,
            specular_tint,
            roughness,
            anisotropic,
            sheen,
            sheen_tint,
            clearcoat,
            clearcoat_gloss,
            ior,
        };

        Self {
            params,
            disney_diffuse: DisneyDiffuse::new_from_texture(
                base_color.clone(),
                roughness,
                subsurface,
            ),
            disney_glass: DisneyGlass::new_from_texture(
                base_color.clone(),
                roughness,
                anisotropic,
                ior,
            ),
            disney_metal: DisneyMetal::new_from_texture(base_color.clone(), roughness, anisotropic),
            disney_sheen: DisneySheen::new_from_texture(base_color.clone(), sheen, sheen_tint),
            disney_clearcoat: DisneyClearcoat::new(clearcoat, clearcoat_gloss),
        }
    }

    fn evaluate_disney(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        rec: &HitRecord,
    ) -> Color {
        // Whether w_o is transmitting
        let transmitted = !rec.front_face();

        let f_diffuse = if transmitted {
            Color::zeros()
        } else {
            self.disney_diffuse.evaluate_diffuse(w_o, h, w_i, rec)
        };
        let f_sheen = if transmitted {
            Color::zeros()
        } else {
            self.disney_sheen.evaluate_sheen(w_o, h, w_i, rec)
        };
        let f_metal = if transmitted {
            Color::zeros()
        } else {
            let base_color = self.params.base_color.value(rec.u(), rec.v(), &rec.p());
            let c_tint = calculate_tint(&base_color);
            let k_s = (1.0 - self.params.specular_tint) * Color::new(1.0, 1.0, 1.0)
                + self.params.specular_tint * c_tint;
            let c_0 =
                self.params.specular * r_0(self.params.ior) * (1.0 - self.params.metallic) * k_s
                    + self.params.metallic * base_color;
            self.disney_metal
                .evaluate_metal_achromatic(w_o, h, w_i, rec, c_0)
        };
        let f_clearcoat = if transmitted {
            Color::zeros()
        } else {
            self.disney_clearcoat.evaluate_clearcoat(w_o, h, w_i, rec)
        };
        let f_glass = self.disney_glass.evaluate_glass(w_o, h, w_i, rec);

        let diffuse =
            (1.0 - self.params.specular_transmission) * (1.0 - self.params.metallic) * f_diffuse;
        let sheen = (1.0 - self.params.metallic) * self.params.sheen * f_sheen;
        let metal =
            (1.0 - (self.params.specular_transmission * (1.0 - self.params.metallic))) * f_metal;
        let clearcoat = 0.25 * self.params.clearcoat * f_clearcoat;
        let glass = (1.0 - self.params.metallic) * self.params.specular_transmission * f_glass;

        diffuse + sheen + metal + clearcoat + glass
    }
}

impl Material for Disney {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        // Compute half normal
        let eta = if rec.front_face() {
            1.0 / self.params.ior
        } else {
            self.params.ior
        };
        let reflect = w_out.y * w_in.y > 0.0;
        let h = if reflect {
            (w_out + w_in).normalize()
        } else {
            (eta * w_out + w_in).normalize()
        };

        self.evaluate_disney(&w_out, &h, &w_in, &rec)
    }

    fn scatter(&self, rng: &mut ThreadRng, w_out: &Vector3<f32>, rec: &HitRecord) -> ScatterRecord {
        // Whether w_out is in transmission
        let transmitted = !rec.front_face();

        // Determine relative weights
        let diffuse_weight = if transmitted {
            0.0
        } else {
            (1.0 - self.params.metallic) * (1.0 - self.params.specular_transmission)
        };
        let metal_weight = if transmitted {
            0.0
        } else {
            1.0 - self.params.specular_transmission * (1.0 - self.params.metallic)
        };
        let glass_weight = (1.0 - self.params.metallic) * self.params.specular_transmission;
        let clearcoat_weight = if transmitted {
            0.0
        } else {
            0.25 * self.params.clearcoat
        };

        let total_weight = diffuse_weight + metal_weight + glass_weight + clearcoat_weight;
        let u1 = rng.gen_range(0.0..total_weight);

        // Sample
        if u1 < diffuse_weight {
            // Diffuse
            self.disney_diffuse.scatter(rng, &w_out, &rec)
        } else if u1 - diffuse_weight < metal_weight {
            // Metal
            self.disney_metal.scatter(rng, &w_out, &rec)
        } else if u1 - (diffuse_weight + metal_weight) < glass_weight {
            // Glass
            self.disney_glass.scatter(rng, &w_out, &rec)
        } else {
            // Clearcoat
            self.disney_clearcoat.scatter(rng, &w_out, &rec)
        }
    }

    fn scattering_pdf(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> f32 {
        let transmitted = !rec.front_face();

        // Determine relative weights
        let diffuse_weight = if transmitted {
            0.0
        } else {
            (1.0 - self.params.metallic) * (1.0 - self.params.specular_transmission)
        };
        let metal_weight = if transmitted {
            0.0
        } else {
            1.0 - self.params.specular_transmission * (1.0 - self.params.metallic)
        };
        let glass_weight = (1.0 - self.params.metallic) * self.params.specular_transmission;
        let clearcoat_weight = if transmitted {
            0.0
        } else {
            0.25 * self.params.clearcoat
        };

        let total_weight = diffuse_weight + metal_weight + glass_weight + clearcoat_weight;

        let pdf_diffuse = diffuse_weight * self.disney_diffuse.scattering_pdf(w_out, w_in, rec);
        let pdf_metal = metal_weight * self.disney_metal.scattering_pdf(w_out, w_in, rec);
        let pdf_glass = glass_weight * self.disney_glass.scattering_pdf(w_out, w_in, rec);
        let pdf_clearcoat =
            clearcoat_weight * self.disney_clearcoat.scattering_pdf(w_out, w_in, rec);

        (pdf_diffuse + pdf_metal + pdf_glass + pdf_clearcoat) / total_weight
    }
}

//============================================================================//

pub struct DisneySheen {
    base_color: TextureRef,
    sheen: f32,      // TODO: texturize
    sheen_tint: f32, // TODO: texturize
}

impl DisneySheen {
    pub fn new(base_color: Color, sheen: f32, sheen_tint: f32) -> Self {
        Self {
            base_color: Arc::new(Solid::new(base_color)),
            sheen,
            sheen_tint,
        }
    }

    pub fn new_from_texture(base_color: TextureRef, sheen: f32, sheen_tint: f32) -> Self {
        Self {
            base_color,
            sheen,
            sheen_tint,
        }
    }

    fn evaluate_sheen(
        &self,
        _w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        rec: &HitRecord,
    ) -> Color {
        if self.sheen <= 0.0 {
            return Vector3::zeros();
        }

        // Acquire base_color
        let base_color = self.base_color.value(rec.u(), rec.v(), &rec.p());

        let h_dot_w_i = h.dot(&w_i);
        let tint = calculate_tint(&base_color);

        self.sheen
            * Color::new(1.0, 1.0, 1.0).lerp(&tint, self.sheen_tint)
            * schlick_weight(h_dot_w_i)
    }
}

impl Material for DisneySheen {
    fn is_specular(&self) -> bool {
        false
    }

    fn is_emissive(&self) -> bool {
        false
    }

    fn scatter(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        _w_out: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> ScatterRecord {
        // Cosine weighted sampling
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        ScatterRecord {
            w_in: Vector3::new(x, y, z),
        }
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize();
        self.evaluate_sheen(&w_out, &h, &w_in, &rec)
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        &w_in.y * INV_PI
    }
}

//============================================================================//

pub struct DisneyClearcoat {
    clearcoat: f32,
    clearcoat_gloss: f32, // TODO: Texturize
    a2: f32,
    a2_invlog2: f32,
}

impl DisneyClearcoat {
    pub fn new(clearcoat: f32, clearcoat_gloss: f32) -> Self {
        let a = (1.0 - clearcoat_gloss) * 0.1 + (clearcoat_gloss * 0.001);
        let a2 = a * a;

        Self {
            clearcoat,
            clearcoat_gloss,
            a2,
            a2_invlog2: 1.0 / a2.log2(),
        }
    }

    fn gtr1(&self, cos_d: f32, a2: f32) -> f32 {
        if a2 >= 1.0 {
            return INV_PI;
        }

        INV_PI * (a2 - 1.0) * self.a2_invlog2 / (1.0 + (a2 - 1.0) * cos_d * cos_d)
    }

    fn separable_smith_ggx_g1(&self, cos_w: f32, a2: f32) -> f32 {
        2.0 / (1.0 + (a2 + (1.0 - a2) * cos_w * cos_w).sqrt())
    }

    fn separable_g(&self, w: &Vector3<f32>) -> f32 {
        let x = 0.25 * w.x;
        let y = w.y;
        let z = 0.25 * w.z;

        2.0 / (1.0 + (((x * x) + (z * z)) / (y * y))).sqrt()
    }

    fn evaluate_clearcoat(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> Color {
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

    fn scatter(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        w_out: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> ScatterRecord {
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

        let n_e = Vector3::new(x, y, z);
        let w_in = reflect(&w_out, &n_e);

        ScatterRecord { w_in }
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

//============================================================================//

pub struct DisneyMetal {
    base_color: TextureRef,
    roughness: f32,
    anisotropic: f32,

    // Precomputed values
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

    pub fn new_from_texture(base_color: TextureRef, roughness: f32, anisotropic: f32) -> Self {
        const A_MIN: f32 = 0.0001;
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let inv_aspect = 1.0 / aspect;

        let a_x = (roughness * roughness * inv_aspect).max(A_MIN);
        let a_z = (roughness * roughness * aspect).max(A_MIN);

        Self {
            base_color,
            roughness,
            anisotropic,
            a_x,
            a_z,
            inv_a_x: 1.0 / a_x,
            inv_a_z: 1.0 / a_z,
        }
    }

    fn ggx_distribution(&self, h: &Vector3<f32>) -> f32 {
        let k = (h.x * h.x * self.inv_a_x * self.inv_a_x)
            + (h.z * h.z * self.inv_a_z * self.inv_a_z)
            + h.y * h.y;
        INV_PI * self.inv_a_x * self.inv_a_z * (1.0 / (k * k))
    }

    fn separable_g(&self, w: &Vector3<f32>) -> f32 {
        let x = self.a_x * w.x;
        let y = w.y;
        let z = self.a_z * w.z;

        2.0 / (1.0 + (1.0 + ((x * x) + (z * z)) / (y * y)).sqrt())
    }

    fn evaluate_metal(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        rec: &HitRecord,
    ) -> Color {
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

    fn evaluate_metal_achromatic(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        _rec: &HitRecord,
        c_0: Color,
    ) -> Color {
        // Common values
        let cos_o = w_o.y.abs().min(1.0);
        let cos_i = w_i.y.abs().min(1.0);
        let cos_d = h.dot(&w_i).abs().min(1.0);
        let base_color = c_0;

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
        if w_in.y < 0.0 {
            return Color::zeros();
        }
        let h = (w_out + w_in).normalize();
        self.evaluate_metal(&w_out, &h, &w_in, &rec)
    }

    fn scatter(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        w_out: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> ScatterRecord {
        // VNDF from Heitz
        // https://jcgt.org/published/0007/04/01/slides.pdf
        // Generate random samples
        let u1: f32 = rng.gen_range(0.0..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);

        // Transform to hemisphere configuration
        let w_hemi = Vector3::new(self.a_x * w_out.x, w_out.y, self.a_z * w_out.z).normalize();

        // Orthonormal basis
        let basis_1 = if w_hemi.y < 0.9999 {
            Vector3::new(0.0, 1.0, 0.0).cross(&w_hemi).normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let basis_2 = w_hemi.cross(&basis_1);

        // Parameterize projected area
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;
        let t1 = r * phi.cos();
        let t2 = r * phi.sin();
        let s = 0.5 * (1.0 + w_hemi.y);
        let t2 = (1.0 - s) * (1.0 - t1 * t1).sqrt() + (s * t2);

        // Project onto hemisphere
        let n_h = (t1 * basis_1)
            + (t2 * basis_2)
            + (1.0 - (t1 * t1) - (t2 * t2)).max(0.0).sqrt() * w_hemi;

        // Normalize into original configuration
        let n_e = Vector3::new(self.a_x * n_h.x, n_h.y.max(0.0), self.a_z * n_h.z).normalize();
        let w_in = reflect(&w_out, &n_e);

        ScatterRecord { w_in }
    }

    fn scattering_pdf(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        let h = (w_out + w_in).normalize();
        let d = self.ggx_distribution(&h);
        let g_o = self.separable_g(&w_out);
        let cos_i = w_in.y.abs().min(1.0);

        (0.25 * d * g_o) / cos_i
    }
}

//============================================================================//

pub struct DisneyGlass {
    base_color: TextureRef,
    roughness: f32,
    anisotropic: f32,
    eta: f32, // internal IOR / external IOR

    // Precomputed values
    a_x: f32,
    a_z: f32,
    inv_a_x: f32,
    inv_a_z: f32,
}

impl DisneyGlass {
    pub fn new(base_color: Color, roughness: f32, anisotropic: f32, eta: f32) -> Self {
        const A_MIN: f32 = 0.0001;
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let inv_aspect = 1.0 / aspect;

        let a_x = (roughness * roughness * inv_aspect).max(A_MIN);
        let a_z = (roughness * roughness * aspect).max(A_MIN);

        Self {
            base_color: Arc::new(Solid::new(base_color)),
            roughness,
            anisotropic,
            eta,
            a_x,
            a_z,
            inv_a_x: 1.0 / a_x,
            inv_a_z: 1.0 / a_z,
        }
    }

    pub fn new_from_texture(
        base_color: TextureRef,
        roughness: f32,
        anisotropic: f32,
        eta: f32,
    ) -> Self {
        const A_MIN: f32 = 0.0001;
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();
        let inv_aspect = 1.0 / aspect;

        let a_x = (roughness * roughness * inv_aspect).max(A_MIN);
        let a_z = (roughness * roughness * aspect).max(A_MIN);

        Self {
            base_color,
            roughness,
            anisotropic,
            eta,
            a_x,
            a_z,
            inv_a_x: 1.0 / a_x,
            inv_a_z: 1.0 / a_z,
        }
    }

    fn ggx_distribution(&self, h: &Vector3<f32>) -> f32 {
        let h2 = h.component_mul(&h);
        let k = (h2.x * self.inv_a_x * self.inv_a_x) + (h2.z * self.inv_a_z * self.inv_a_z) + h2.y;
        INV_PI * self.inv_a_x * self.inv_a_z * (1.0 / (k * k))
    }

    fn separable_g(&self, w: &Vector3<f32>) -> f32 {
        let x = self.a_x * w.x;
        let y = w.y;
        let z = self.a_z * w.z;

        2.0 / (1.0 + (1.0 + ((x * x) + (z * z)) / (y * y)).sqrt())
    }

    fn evaluate_glass(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        rec: &HitRecord,
    ) -> Color {
        let eta = if rec.front_face() {
            1.0 / self.eta
        } else {
            self.eta
        };
        let reflect = w_o.y * w_i.y > 0.0;

        // Common values
        let cos_o = w_o.y.abs().min(1.0);
        let cos_i = w_i.y.abs().min(1.0);
        let h_dot_o = h.dot(&w_o).clamp(-1.0, 1.0);
        let h_dot_i = h.dot(&w_i).clamp(-1.0, 1.0);
        let base_color = self.base_color.value(rec.u(), rec.v(), &rec.p());

        // Compute Cook-Torrance terms
        let f = fresnel_o(&w_o, &h, eta);
        let d = self.ggx_distribution(&h);
        let g = self.separable_g(&w_o) * self.separable_g(&w_i);

        // Determine whether transmit or to reflect
        if reflect {
            // Reflection

            ((0.25 * f * d * g) / (cos_o * cos_i)) * base_color
        } else {
            // Transmission
            let sqrt_denom = h_dot_o + eta * h_dot_i;
            let base_color = Vector3::new(
                base_color.x.sqrt(),
                base_color.y.sqrt(),
                base_color.z.sqrt(),
            );

            let val = ((1.0 - f) * d * g * (h_dot_i * h_dot_o).abs())
                / (cos_o * cos_i * sqrt_denom * sqrt_denom);

            // log::info!("{f} {d} {g} {val}");
            val * base_color
        }
    }
}

impl Material for DisneyGlass {
    fn is_emissive(&self) -> bool {
        false
    }

    fn is_specular(&self) -> bool {
        false
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        // Compute half normal
        let eta = if rec.front_face() {
            1.0 / self.eta
        } else {
            self.eta
        };
        let reflect = w_out.y * w_in.y > 0.0;
        let h = if reflect {
            (w_out + w_in).normalize()
        } else {
            (eta * w_out + w_in).normalize()
        };

        // Reflect if below surface
        let h = if h.y < 0.0 { -h } else { h };

        self.evaluate_glass(&w_out, &h, &w_in, &rec)
    }

    fn scatter(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        w_out: &Vector3<f32>,
        rec: &HitRecord,
    ) -> ScatterRecord {
        // VNDF from Heitz
        // https://jcgt.org/published/0007/04/01/slides.pdf
        // Generate random samples
        let u1: f32 = rng.gen_range(0.0..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);

        // Transform to hemisphere configuration
        let w_hemi = Vector3::new(self.a_x * w_out.x, w_out.y, self.a_z * w_out.z).normalize();

        // Orthonormal basis
        let basis_1 = if w_hemi.y < 0.9999 {
            Vector3::new(0.0, 1.0, 0.0).cross(&w_hemi).normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let basis_2 = w_hemi.cross(&basis_1);

        // Parameterize projected area
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;
        let t1 = r * phi.cos();
        let t2 = r * phi.sin();
        let s = 0.5 * (1.0 + w_hemi.y);
        let t2 = (1.0 - s) * (1.0 - t1 * t1).sqrt() + (s * t2);

        // Project onto hemisphere
        let n_h = (t1 * basis_1)
            + (t2 * basis_2)
            + (1.0 - (t1 * t1) - (t2 * t2)).max(0.0).sqrt() * w_hemi;

        // Normalize into original configuration
        let n_e = Vector3::new(self.a_x * n_h.x, n_h.y.max(0.0), self.a_z * n_h.z).normalize();
        let eta = if rec.front_face() {
            1.0 / self.eta
        } else {
            self.eta
        };

        let u3: f32 = rng.gen_range(0.0..1.0);

        let fresnel = fresnel_o(&w_out, &n_e, eta);

        // log::info!("{fresnel} {schlick}");

        if fresnel >= u3 {
            let w_in = reflect(&w_out, &n_e);

            ScatterRecord { w_in }
        } else {
            match refract(&w_out, &n_e, eta) {
                Some(w_in) => ScatterRecord { w_in },
                None => {
                    let w_in = reflect(&w_out, &n_e);

                    ScatterRecord { w_in }
                }
            }
        }
    }

    fn scattering_pdf(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> f32 {
        let eta = if rec.front_face() {
            1.0 / self.eta
        } else {
            self.eta
        };
        if w_out.y * w_in.y > 0.0 {
            // Reflection
            let h = (w_out + w_in).normalize();
            let d = self.ggx_distribution(&h);
            let g_o = self.separable_g(&w_out);
            let cos_o = w_out.y.abs().min(1.0);
            let fresnel = fresnel_o(&w_out, &h, eta);
            (0.25 * fresnel * d * g_o) / cos_o
        } else {
            // Refraction
            let h = (eta * w_out + w_in).normalize();
            let h = if h.y < 0.0 { -h } else { h };
            let fresnel = fresnel_o(&w_out, &h, eta);
            let d = self.ggx_distribution(&h);
            let g_o = self.separable_g(&w_out);

            let h_dot_i = h.dot(&w_in);
            let h_dot_o = h.dot(&w_out);
            let sqrt_denom = h_dot_o + eta * h_dot_i;
            let dh_di = eta * eta * h_dot_i / (sqrt_denom * sqrt_denom);
            let cos_o = w_out.y.abs().min(1.0);

            (1.0 - fresnel) * d * g_o * (dh_di * h_dot_o / (cos_o)).abs()
        }
    }
}

//============================================================================//

pub struct DisneyDiffuse {
    base_color: TextureRef,
    roughness: f32,  // TODO: Texturize
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

    pub fn new_from_texture(base_color: TextureRef, roughness: f32, subsurface: f32) -> Self {
        Self {
            base_color,
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

    fn evaluate_diffuse(
        &self,
        w_o: &Vector3<f32>,
        h: &Vector3<f32>,
        w_i: &Vector3<f32>,
        rec: &HitRecord,
    ) -> Color {
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

    fn scatter(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        _w_out: &Vector3<f32>,
        _rec: &HitRecord,
    ) -> ScatterRecord {
        // Cosine weighted sampling
        let r1: f32 = rng.gen_range(0.0..1.0);
        let r2: f32 = rng.gen_range(0.0..1.0);

        let phi = 2.0 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2.sqrt();
        let z = phi.sin() * r2.sqrt();
        let y = (1.0 - r2).sqrt();

        ScatterRecord {
            w_in: Vector3::new(x, y, z),
        }
    }

    fn bsdf_evaluate(&self, w_out: &Vector3<f32>, w_in: &Vector3<f32>, rec: &HitRecord) -> Color {
        let h = (w_out + w_in).normalize();
        self.evaluate_diffuse(&w_out, &h, &w_in, &rec)
    }

    fn scattering_pdf(&self, _w_out: &Vector3<f32>, w_in: &Vector3<f32>, _rec: &HitRecord) -> f32 {
        &w_in.y * INV_PI
    }
}
