
# pathtracer

Toy project to learn rust

![cool render](day3.png)
Current progress (A cornell box, 600x600, 10240 spp, 1061s)

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
- Support loading/writing materials, objects, etc from config files
- instance contents currently aren't bvh'ed.
- lol i just realized vtables r probably cache thrashing

TODO LATER: 
- gui
- make faster
    - tiling
    - for larger meshes consider doing lbvh
- proper brdf
- variance reduction
- better pathtracing methods
    - bidirectional
    - ma la tang

TODO LATER LATER:
- tests haha


[_Ray Tracing: The Next Week_](https://raytracing.github.io/books/RayTracingTheNextWeek.html)

[PBR Book](https://www.pbr-book.org/4ed/contents)
