
# pathtracer

Toy project to learn rust

![cool render](day3.png)

Current progress (mitsuba knob, 768x768, 10240 spp, 2653s)

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

TODO: 
- env maps
- sampling for triangles
- importance sampling
- Support loading/writing materials, objects, etc from config files
- instance contents currently aren't bvh'ed.

TODO LATER: 
- gui
- make faster
- better pathtracing methods
    - bidirectional
    - ma la tang

TODO LATER LATER:
- tests haha


[_Ray Tracing: The Next Week_](https://raytracing.github.io/books/RayTracingTheNextWeek.html)

[PBR Book](https://www.pbr-book.org/4ed/contents)
