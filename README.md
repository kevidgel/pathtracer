
# pathtracer

Toy project to learn rust

I don't know if I should continue with rust or move on to using gpu...

![cool render](day3.png)

Current progress (mitsuba knob with Disney material, 768x768, 10240 spp, ~1500s)

To render the Cornell box scene:

Build:
```shell
cargo build --release
```
Run:
```shell
./target/release/pathtracer
```

Scene configurations are located under `scenes`

IMPLEMENTED:
- All of raytracing.github.io
- Flattened BVH
- Disney BSDF
- Loading of .obj files
- Programmatic scene description (see /scenes)

TODO:
- .glb files
- Use a thread pool to reduce thread spawn/destroy overhead
- light sampling seems incorrect? (probably due to specular materials not being handled properly)
- Disney material is a little incorrect
- env maps
- correct sampling for triangles
- multiple importance sampling
- support some standard scene description format (.glb)

TODO LATER: 
- gui
- make faster
- better pathtracing methods
    - bidirectional
    - metropolis light transport

TODO LATER LATER:
- tests haha


[_Ray Tracing: The Next Week_](https://raytracing.github.io/books/RayTracingTheNextWeek.html)

[PBR Book](https://www.pbr-book.org/4ed/contents)
