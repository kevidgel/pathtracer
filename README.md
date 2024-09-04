
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

TODO:
- light sampling seems incorrect?
- Disney material is a little incorrect
- env maps
- correct sampling for triangles
- multiple importance sampling
- Support loading/writing materials, objects, etc from config files

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
