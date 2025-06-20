# WWINDOW: A WebAssembly friendly window and rendering using Rust

To address different windowing, rendering, and surfacing approaches between web and native:

- Window Management: Winit creates and manages windows, handling platform-specific details for desktop and web environments
- Event Loop: Provides a unified event system for user input (keyboard, mouse, touch), window events (resize, focus, close), and system events
- Thin Networking Layer: Wry allows an embeddable webview on native to effectively enable networking in a K.I.S.S. format, while fetch() is available for the browser version as a web standard via js_sys / web_sys.
- Platform Abstraction: Seamlessly abstracts OS-specific windowing APIs behind a consistent interface

## Platform Implementations

Unlike many window management libraries, winit has first-class web support through wry, which provides a native web view, and web_sys & js_sys for browser WASM contexts. It translates web events (mouse, keyboard, touch, gamepad) into a unified event system that works identically across all platforms. Winit implements the raw_window_handle::HasWindowHandle and raw_window_handle::HasDisplayHandle traits under the hood, which is perfect for our usecase.

- Wry: Accepts Winit's Raw Window Handle Implementations for Native Chromium WebView
  - Windows: Provides HWND handle for Win32 windows
  - macOS: Provides NSWindow handle for Cocoa windows
  - Linux: Provides X11 Window ID or Wayland surface handle
  - Web/WASM: Provides HTML canvas element handle
    - Integrated Networking via web_sys, js_sys fetch()

The graphics library (like WGPU) takes Winit's Raw Window Handle and creates the platform-appropriate surface (Direct3D/Vulkan on Windows, Metal on macOS, Vulkan on Linux, WebGPU on web).

The overhead is front-loaded during initialization, then it gets out of the way for actual rendering. Use #[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]

## Graphics API - WGPU

- Minimal overhead - it's a thin wrapper around platform APIs
- Efficient & Transparent
- No hidden memory costs or background threads

- Resolution: 1080x720 (Support for 1920x1080 TBD)
- Target FPS: 60+
- Quality: AA/AAA-tier
- Platform: Web + Native

- API: WebGPU via wgpu (web + native)
- Window: winit with raw_window_handle
- Surface: Platform-appropriate (HWND/NSWindow/X11/Canvas)

## Usage Notes

- WebGPU on Windows with Chrome requires you to set your preferred device to the high performance method in you GPU manager and in the windows graphics settings.
- You will also need to enable experimental webgpu flags in chrome.

## WGPU Implementation

This configuration ensures maximum compatibility across platforms while maintaining simplicity.

```toml
[dependencies]
winit = "0.30.8"
wgpu = { version = "25.0.0", features = ["webgpu"]}
wry = "0.51.2"
```

```rust
cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let limits = wgpu::Limits::downlevel_webgl2_defaults();
    } else {
        let instance = wgpu::Instance::default();
        let limits = wgpu::Limits::default();
    }
}
```

# 3D Engine Implementation Guide

## Overview

This document covers the implementation of 3D rendering capabilities using `glam` and `bytemuck` libraries in a WASM-compatible 3D engine built on `wgpu`.

## Dependencies

```toml
[dependencies]
glam = "0.24"
bytemuck = { version = "1.13", features = ["derive"] }
wgpu = { version = "0.20", features = ["webgl"] }
```

## Core Components

### 1. Vertex Structure

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}
```

- Uses `bytemuck` traits for safe GPU memory layout
- `#[repr(C)]` ensures predictable memory layout
- Position and color data per vertex

### 2. Uniform Buffer

```rust
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],  // 4x4 matrix as 2D array
}
```

- Stores transformation matrices for GPU shaders
- `glam::Mat4` converted to GPU-compatible format via `to_cols_array_2d()`

### 3. Geometry Data

#### Cube Vertices (24 vertices for 6 faces)

```rust
let vertices = &[
    // Front face (red) - 4 vertices
    // Back face (green) - 4 vertices
    // Left face (blue) - 4 vertices
    // Right face (yellow) - 4 vertices
    // Top face (magenta) - 4 vertices
    // Bottom face (cyan) - 4 vertices
];
```

#### Index Buffer

```rust
let indices: &[u16] = &[
    0,1,2, 2,3,0,   // front (2 triangles)
    4,5,6, 6,7,4,   // back
    8,9,10, 10,11,8, // left
    12,13,14, 14,15,12, // right
    16,17,18, 18,19,16, // top
    20,21,22, 22,23,20, // bottom
];
```

- Each face = 2 triangles = 6 indices
- Uses `u16` indices with `IndexFormat::Uint16`

## 3D Transformation Pipeline

### Matrix Transformation Order

```rust
let model_view_proj = proj_matrix * view_matrix * rotation_matrix;
```

**Pipeline**: Object Space → World Space → View Space → Clip Space → Screen Space

### 1. Model Matrix (Object Transform)

```rust
let rotation_matrix = Mat4::from_rotation_y(self.rotation) * Mat4::from_rotation_x(self.rotation * 0.7);
```

- **Purpose**: Rotates your cube in object space
- **Y-rotation**: Horizontal spinning
- **X-rotation**: Vertical tilting
- **Multiplier 0.7**: Creates varied rotation speeds

CRITICAL: Matrix multiplication order matters! This reads right-to-left:

rotation_matrix: Transform cube vertices (object → world space)
view_matrix: Transform relative to camera (world → view space)
proj_matrix: Apply perspective (view → clip space)

### 2. View Matrix (Camera)

```rust
let eye = Vec3::new(4.0, 3.0, 2.0);     // Camera position
let target = Vec3::ZERO;                // Look-at point
let up = Vec3::Y;                       // Up direction
let view_matrix = Mat4::look_at_rh(eye, target, up);
```

What this does:

eye: Places your camera at position (4, 3, 2) - slightly above, to the right, and forward
target: Camera looks at the origin (0, 0, 0) where your cube is
up: Tells the camera that the Y-axis points "up" (prevents rolling)
look_at_rh: "Right-handed" coordinate system (common in graphics)

Think of this as moving the world relative to your camera. If you move the camera right, the world appears to move left.

### 3. Projection Matrix (Perspective)

```rust
let aspect = width as f32 / height as f32;
let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
```

What this does:

45.0_f32.to_radians(): Field of view angle (45 degrees) - how "wide" the camera sees
aspect: Width/height ratio to prevent stretching on non-square screens
0.1: Near clipping plane - anything closer than 0.1 units gets cut off
100.0: Far clipping plane - anything farther than 100 units gets cut off

This creates the perspective effect where distant objects appear smaller.

### 4. Matrix Multiplication Order (Shader Usage)

```rust
let model_view_proj = proj_matrix _ view_matrix _ rotation_matrix;
```

CRITICAL: Matrix multiplication order matters! This reads right-to-left:

rotation_matrix: Transform cube vertices (object → world space)
view_matrix: Transform relative to camera (world → view space)
proj_matrix: Apply perspective (view → clip space)

## Shader Implementation

### WGSL Vertex Shader

```wgsl
@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}
```

### WGSL Fragment Shader

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

## Rendering Pipeline

### Buffer Creation

```rust
let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("Vertex Buffer"),
    contents: bytemuck::cast_slice(vertices),
    usage: wgpu::BufferUsages::VERTEX,
});
```

### Depth Testing

```rust
depth_stencil: Some(wgpu::DepthStencilState {
    format: wgpu::TextureFormat::Depth32Float,
    depth_write_enabled: true,
    depth_compare: wgpu::CompareFunction::Less,
    // ...
})
```

### Per-Frame Updates

```rust
// Update rotation
self.rotation += 0.01;

// Recreate transformation matrix
let model_view_proj = proj_matrix * view_matrix * rotation_matrix;
self.uniforms.update_view_proj(model_view_proj);

// Upload to GPU
self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
```

## Key Integration Points

### State Structure Extensions

```rust
pub struct State {
    // ... existing wgpu fields ...
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    rotation: f32,
}
```

### Required Imports

```rust
use glam::{Mat4, Vec3, Vec4};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
```

## Alternative Rotation Patterns

```rust
// Horizontal spin only
let rotation_matrix = Mat4::from_rotation_y(self.rotation);

// Chaotic tumbling (3 axes)
let rotation_matrix = Mat4::from_rotation_y(self.rotation)
    * Mat4::from_rotation_x(self.rotation * 0.7)
    * Mat4::from_rotation_z(self.rotation * 0.3);

// Diagonal axis rotation
let diagonal_axis = Vec3::new(1.0, 1.0, 1.0).normalize();
let rotation_matrix = Mat4::from_axis_angle(diagonal_axis, self.rotation);

// Rotating camera
let camera_angle = self.rotation * 0.5;
let eye = Vec3::new(
    4.0 * camera_angle.cos(),
    3.0,
    4.0 * camera_angle.sin()
);
```

## Common Issues & Solutions

### Index Buffer Format Mismatch

- **Problem**: Using `&[u32]` indices with `IndexFormat::Uint16`
- **Solution**: Use `&[u16]` indices OR change to `IndexFormat::Uint32`

### Entry Point Wrapping

- **Problem**: `entry_point: "vs_main"` in newer wgpu versions
- **Solution**: `entry_point: Some("vs_main")`

### Matrix Order

- **Problem**: Incorrect transformation order
- **Solution**: Remember right-to-left: `projection * view * model`

## Performance Notes

- Use `bytemuck` for zero-copy vertex data casting
- Update uniform buffers only when transformation changes
- Consider instancing for multiple objects
- Depth testing essential for correct 3D rendering
