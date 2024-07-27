use config::{Config, ConfigError};

pub type Result<T> = std::result::Result<T, ConfigError>;

pub fn read_config(config_path: &str) -> Result<Config> {
    let full_path = format!("src/configs/{}", config_path);
    let cfg = Config::builder()
        .add_source(config::File::with_name(full_path.as_str()))
        .build()?;

    Ok(cfg)
}

pub trait FromConfig {
    fn build(cfg: Config) -> Self;
}