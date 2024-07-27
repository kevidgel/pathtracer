use toml::Value;

struct PTConfig {
    cameras: Option<Vec<Value>>,
    objects: Option<Vec<Value>>,
    materials: Option<Vec<Value>>,
}
