use image::Rgb;

pub type Color = na::Vector3<f32>;

pub trait ColorOps {
    fn gray(val: f32) -> Self;
    fn get_r(&self) -> f32;
    fn get_g(&self) -> f32;
    fn get_b(&self) -> f32;
    fn to_u32(&self) -> u32;
    fn to_rgb(&self) -> Rgb<u8>;
    fn linear_to_gamma(c: f32) -> f32;
    fn get_r_gamma(&self) -> f32;
    fn get_g_gamma(&self) -> f32;
    fn get_b_gamma(&self) -> f32;
    fn random() -> Self;
    fn random_range(min: f32, max: f32) -> Self;
}

impl ColorOps for Color {
    fn gray(val: f32) -> Self {
        Color::new(val, val, val)
    }
    fn get_r(&self) -> f32 {
        self.x
    }

    fn get_g(&self) -> f32 {
        self.y
    }

    fn get_b(&self) -> f32 {
        self.z
    }

    fn linear_to_gamma(c: f32) -> f32 {
        if c < 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    fn get_r_gamma(&self) -> f32 {
        Self::linear_to_gamma(self.get_r())
    }

    fn get_g_gamma(&self) -> f32 {
        Self::linear_to_gamma(self.get_g())
    }

    fn get_b_gamma(&self) -> f32 {
        Self::linear_to_gamma(self.get_b())
    }

    fn to_u32(&self) -> u32 {
        let mut ir: u32 = (255.999 * self.get_r_gamma()) as u32;
        let mut ig: u32 = (255.999 * self.get_g_gamma()) as u32;
        let mut ib: u32 = (255.999 * self.get_b_gamma()) as u32;

        ir = ir.clamp(0, 255);
        ig = ig.clamp(0, 255);
        ib = ib.clamp(0, 255);

        ir << 16 | ig << 8 | ib
    }

    fn to_rgb(&self) -> Rgb<u8> {
        let mut ir: u32 = (255.999 * self.get_r_gamma()) as u32;
        let mut ig: u32 = (255.999 * self.get_g_gamma()) as u32;
        let mut ib: u32 = (255.999 * self.get_b_gamma()) as u32;

        ir = ir.clamp(0, 255);
        ig = ig.clamp(0, 255);
        ib = ib.clamp(0, 255);

        Rgb([ir as u8, ig as u8, ib as u8])
    }

    fn random() -> Self {
        Color::new(
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        )
    }

    fn random_range(min: f32, max: f32) -> Self {
        Color::new(
            rand::random::<f32>() * (max - min) + min,
            rand::random::<f32>() * (max - min) + min,
            rand::random::<f32>() * (max - min) + min,
        )
    }
}
