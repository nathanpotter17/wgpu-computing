# WCOMPUTE: Sample Limit of Ray-based Rendering Pipeline on WGSL using WGPU

To sample the limitations of ray based compute shading for a web target:

- Example of pure ray based rendering & shading using compute only and its limitations on web targets, especially not ideal for irregular sized scenes.
- Simple Blinn Phong lighting model, complex scene (125 objects) can only render at 50FPS on a 256x256 resolution.
- Performance hard-capped to 50FPS on web via presentation mode availibilty. [WGPU Presentation Modes and Limits](https://docs.rs/wgpu/25.0.2/wgpu/enum.PresentMode.html)
- [WGPU Healess Example](https://github.com/sotrh/learn-wgpu/blob/master/code/showcase/windowless/src/main.rs)
