use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, MouseButton as WinitMouseButton},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
    keyboard::{PhysicalKey, KeyCode as WinitKeyCode},
};

use glam::{Mat4, Vec3, Vec4};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use bitflags::bitflags;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// === CONSTANTS ===
const DIMX: u32 = 1080;
const DIMY: u32 = 720;
const TARGET_FPS: f32 = 512.0;
const FRAME_TIME: f32 = 1.0 / TARGET_FPS;
const MINIMAL_LOGGING: bool = true;
const LOGGING_ENABLED: bool = false;
const LOGGING_TIMESTEP: u32 = 240;
const STATS_UPDATE_INTERVAL: f32 = 0.75;
const MAX_LIGHTS: usize = 8;
const CAMERA_MOVE_SPEED: f32 = 5.0;
const FOV_STEP: f32 = 5.0;
const LIGHT_INTENSITY_STEP: f32 = 0.5;
const LIGHT_ANIMATION_RADIUS: f32 = 15.0;
const LIGHT_ANIMATION_HEIGHT: f32 = 10.0;
const LIGHT_ANIMATION_SPEED: f32 = 1.0;
const CAMERA_ROTATION_SPEED_X: f32 = 0.42;
const CAMERA_ROTATION_SPEED_Y: f32 = 0.6;
const MAX_INSTANCES: usize = 256;
const FRUSTUM_CULL_ENABLED: bool = true;
const COMPUTE_WORKGROUP_SIZE: u32 = 16;
const LIGHTING_TILE_SIZE: u32 = 16;
const DIRTY_CAMERA: u8 = 1 << 0;
const DIRTY_LIGHTS: u8 = 1 << 1;
const DIRTY_INSTANCES: u8 = 1 << 2;

// ====================
// === INPUT SYSTEM ===
// ====================

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct InputFlags: u32 {
        const W = 1 << 0;
        const A = 1 << 1;
        const S = 1 << 2;
        const D = 1 << 3;
        const Q = 1 << 4;
        const E = 1 << 5;
        const R = 1 << 6;
        const L = 1 << 7;
        const P = 1 << 8;
        const I = 1 << 9;
        const TAB = 1 << 10;
        const ESC = 1 << 11;
        const DIGIT1 = 1 << 12;
        const DIGIT2 = 1 << 13;
        const ARROW_UP = 1 << 14;
        const ARROW_DOWN = 1 << 15;
        const ARROW_LEFT = 1 << 16;
        const ARROW_RIGHT = 1 << 17;
        const MOUSE_LEFT = 1 << 18;
        const MOUSE_RIGHT = 1 << 19;
        const MOUSE_MIDDLE = 1 << 20;
        const C = 1 << 21;
        const SPACE = 1 << 22;
        const F = 1 << 23;
        const MOVEMENT = Self::W.bits() | Self::A.bits() | Self::S.bits() | Self::D.bits() | Self::Q.bits() | Self::E.bits() | Self::SPACE.bits();
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct StateFlags: u32 {
        const RUNNING = 1 << 0;
        const PAUSED = 1 << 1;
        const LIGHT_ANIMATION = 1 << 2;
        const CAMERA_AUTO_ROTATE = 1 << 3;
        const SHOULD_EXIT = 1 << 4;
        const USE_COMPUTE_LIGHTING = 1 << 5;
        const FRUSTUM_CULLING = 1 << 7;
    }
}

// ======================================
// === SHADER DATA STRUCTURES ===
// ======================================

#[repr(C, align(64))]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct LightData {
    position: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    attenuation_constant: f32,
    attenuation_linear: f32,
    attenuation_quadratic: f32,
    _padding: [f32; 6],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 3],
    light_count: u32,
    time: f32,
    frame_index: u32,
    instance_count: u32,
    _padding: [f32; 1],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct LightBuffer {
    lights: [LightData; MAX_LIGHTS],
}

// Enhanced instance data for SDF primitives
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct InstanceData {
    model_matrix: [[f32; 4]; 4],
    color: [f32; 4],            // xyz = color, w = metallic
    material: [f32; 4],          // x = roughness, y = ao, z = emission strength, w = sdf type
    sdf_params: [f32; 4],        // SDF-specific parameters (e.g., radius, size, rounding)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ComputeUniforms {
    screen_dims: [u32; 2],
    tile_counts: [u32; 2],
    time: f32,
    delta_time: f32,
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// ======================================
// === CORE UNIFIED SYSTEM ===
// ======================================

pub struct UnifiedCore {
    pub transform: TransformSystem,
    pub render: RenderSystem,
    pub compute: ComputeSystem,
    pub input: InputSystem,
    pub timing: TimingSystem,
    pub state_flags: StateFlags,
    pub stats_accumulator: StatsAccumulator,
}

#[repr(C, align(64))]
pub struct TransformSystem {
    pub view_matrix: Mat4,
    pub proj_matrix: Mat4,
    pub view_proj_matrix: Mat4,
    pub camera_position: Vec3,
    pub camera_target: Vec3,
    pub camera_rotation: Vec3,
    pub camera_fov: f32,
    pub camera_aspect: f32,
    pub lights: [LightData; MAX_LIGHTS],
    pub light_count: u32,
    pub animation_time: f32,
    pub dirty_flags: u8,
}

pub struct RenderSystem {
    pub uniform_buffer: Option<wgpu::Buffer>,
    pub light_buffer: Option<wgpu::Buffer>,
    pub instance_buffer: Option<wgpu::Buffer>,
    pub render_pipeline: Option<wgpu::RenderPipeline>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub num_indices: u32,
    pub instances: Vec<InstanceData>,
    pub visible_instances: Vec<u32>,
}

pub struct ComputeSystem {
    pub enabled: bool,
    pub compute_pipeline: Option<wgpu::ComputePipeline>,
    pub compute_bind_group: Option<wgpu::BindGroup>,
    pub compute_uniform_buffer: Option<wgpu::Buffer>,
    pub lighting_buffer: Option<wgpu::Buffer>,
    pub tile_count_x: u32,
    pub tile_count_y: u32,
}

pub struct InputSystem {
    pub current_keys: InputFlags,
    pub prev_keys: InputFlags,
    pub mouse_pos: (f32, f32),
    pub mouse_delta: (f32, f32),
    pub mouse_captured: bool,
    pub pointer_lock_frames: u8,
    pub centering_camera: bool,
    pub pointer_lock_transition: bool,
}

pub struct TimingSystem {
    #[cfg(not(target_arch = "wasm32"))]
    pub start: Instant,
    #[cfg(target_arch = "wasm32")]
    pub start_time_ms: f64,
    pub last_frame_time: f32,
    pub delta_time: f32,
    pub accumulator: f32,
    pub frame_count: u32,
}

pub struct StatsAccumulator {
    pub frame_times: [f32; 60],
    pub frame_index: usize,
    pub update_timer: f32,
    #[cfg(target_arch = "wasm32")]
    pub stats_buffer: String,
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self {
            frame_times: [0.0; 60],
            frame_index: 0,
            update_timer: 0.0,
            #[cfg(target_arch = "wasm32")]
            stats_buffer: String::new(),
        }
    }
}

// ============================
// === SHADER SOURCES ===
// ============================

fn generate_render_shader(lighting_tile_size: u32, max_lights: usize, max_instances: usize) -> String {
    format!(r#"
    const MAX_MARCHING_STEPS: i32 = 128;
    const MIN_DIST: f32 = 0.0;
    const MAX_DIST: f32 = 100.0;
    const EPSILON: f32 = 0.0001;
    const SHADOW_EPSILON: f32 = 0.001;
    const AO_SAMPLES: i32 = 5;
    const AO_STEP: f32 = 0.1;
    const MAX_INSTANCES: u32 = {max_instances}u;

    struct VertexInput {{
        @location(0) position: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @location(2) uv: vec2<f32>,
    }}

    struct VertexOutput {{
        @builtin(position) clip_position: vec4<f32>,
        @location(0) ray_dir: vec3<f32>,
        @location(1) uv: vec2<f32>,
    }}

    struct Uniforms {{
        view_proj: mat4x4<f32>,
        camera_position: vec3<f32>,
        light_count: u32,
        time: f32,
        frame_index: u32,
        instance_count: u32,
    }}

    struct LightData {{
        position: vec3<f32>,
        intensity: f32,
        color: vec3<f32>,
        attenuation_constant: f32,
        attenuation_linear: f32,
        attenuation_quadratic: f32,
        _padding0: f32,
        _padding1: f32,
        _padding2: f32,
        _padding3: f32,
        _padding4: f32,
        _padding5: f32,
    }}

    struct LightBuffer {{
        lights: array<LightData, {max_lights}>,
    }}

    struct TileLightData {{
        light_indices: array<u32, {max_lights}>,
        light_count: u32,
    }}

    struct ComputeUniforms {{
        screen_dims: vec2<u32>,
        tile_counts: vec2<u32>,
        time: f32,
        delta_time: f32,
    }}

    struct InstanceData {{
        model_matrix: mat4x4<f32>,
        color: vec4<f32>,
        material: vec4<f32>,
        sdf_params: vec4<f32>,
    }}

    struct InstanceBuffer {{
        instances: array<InstanceData, {max_instances}>,
    }}

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var<uniform> light_buffer: LightBuffer;
    @group(0) @binding(2) var<storage, read> tile_lights: array<TileLightData>;
    @group(0) @binding(3) var<uniform> compute_uniforms: ComputeUniforms;
    @group(0) @binding(4) var<storage, read> instance_buffer: InstanceBuffer;

    // SDF primitives
    fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {{
        return length(p) - radius;
    }}

    fn sdf_box(p: vec3<f32>, size: vec3<f32>) -> f32 {{
        let q = abs(p) - size;
        return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }}

    fn sdf_rounded_box(p: vec3<f32>, size: vec3<f32>, radius: f32) -> f32 {{
        let q = abs(p) - size;
        return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - radius;
    }}

    fn sdf_torus(p: vec3<f32>, radius_major: f32, radius_minor: f32) -> f32 {{
        let q = vec2<f32>(length(p.xz) - radius_major, p.y);
        return length(q) - radius_minor;
    }}

    fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, distance: f32) -> f32 {{
        return dot(p, normal) + distance;
    }}

    fn transform_point(p: vec3<f32>, inv_transform: mat4x4<f32>) -> vec3<f32> {{
        let transformed = inv_transform * vec4<f32>(p, 1.0);
        return transformed.xyz;
    }}

    fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {{
        let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
        let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
        let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
        let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

        let b00 = a00 * a11 - a01 * a10;
        let b01 = a00 * a12 - a02 * a10;
        let b02 = a00 * a13 - a03 * a10;
        let b03 = a01 * a12 - a02 * a11;
        let b04 = a01 * a13 - a03 * a11;
        let b05 = a02 * a13 - a03 * a12;
        let b06 = a20 * a31 - a21 * a30;
        let b07 = a20 * a32 - a22 * a30;
        let b08 = a20 * a33 - a23 * a30;
        let b09 = a21 * a32 - a22 * a31;
        let b10 = a21 * a33 - a23 * a31;
        let b11 = a22 * a33 - a23 * a32;

        let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        let inv_det = 1.0 / det;

        return mat4x4<f32>(
            vec4<f32>(
                (a11 * b11 - a12 * b10 + a13 * b09) * inv_det,
                (a02 * b10 - a01 * b11 - a03 * b09) * inv_det,
                (a31 * b05 - a32 * b04 + a33 * b03) * inv_det,
                (a22 * b04 - a21 * b05 - a23 * b03) * inv_det
            ),
            vec4<f32>(
                (a12 * b08 - a10 * b11 - a13 * b07) * inv_det,
                (a00 * b11 - a02 * b08 + a03 * b07) * inv_det,
                (a32 * b02 - a30 * b05 - a33 * b01) * inv_det,
                (a20 * b05 - a22 * b02 + a23 * b01) * inv_det
            ),
            vec4<f32>(
                (a10 * b10 - a11 * b08 + a13 * b06) * inv_det,
                (a01 * b08 - a00 * b10 - a03 * b06) * inv_det,
                (a30 * b04 - a31 * b02 + a33 * b00) * inv_det,
                (a21 * b02 - a20 * b04 - a23 * b00) * inv_det
            ),
            vec4<f32>(
                (a11 * b07 - a10 * b09 - a12 * b06) * inv_det,
                (a00 * b09 - a01 * b07 + a02 * b06) * inv_det,
                (a31 * b01 - a30 * b03 - a32 * b00) * inv_det,
                (a20 * b03 - a21 * b01 + a22 * b00) * inv_det
            )
        );
    }}

    // Material properties
    struct Material {{
        albedo: vec3<f32>,
        metallic: f32,
        roughness: f32,
        ao: f32,
        emission: vec3<f32>,
    }}

    // Scene SDF with instanced primitives
    fn scene_sdf(p: vec3<f32>) -> vec2<f32> {{
        var min_dist = MAX_DIST;
        var material_id = 0.0;

        // Floor plane
        let floor_dist = sdf_plane(p, vec3<f32>(0.0, 1.0, 0.0), 0.1);
        if (floor_dist < min_dist) {{
            min_dist = floor_dist;
            material_id = 0.0;
        }}

        // Evaluate instanced SDF primitives
        let instance_count = min(uniforms.instance_count, MAX_INSTANCES);
        for (var i = 0u; i < instance_count; i++) {{
            let instance = instance_buffer.instances[i];
            let sdf_type = instance.material.w;
            
            // Transform point to object space
            let inv_transform = inverse_mat4(instance.model_matrix);
            let local_p = transform_point(p, inv_transform);
            
            var dist = MAX_DIST;
            
            // Evaluate SDF based on type
            if (sdf_type < 0.5) {{
                // Sphere
                dist = sdf_sphere(local_p, instance.sdf_params.x);
            }} else if (sdf_type < 1.5) {{
                // Box
                dist = sdf_box(local_p, instance.sdf_params.xyz);
            }} else if (sdf_type < 2.5) {{
                // Rounded box
                dist = sdf_rounded_box(local_p, instance.sdf_params.xyz, instance.sdf_params.w);
            }} else if (sdf_type < 3.5) {{
                // Torus
                dist = sdf_torus(local_p, instance.sdf_params.x, instance.sdf_params.y);
            }}
            
            if (dist < min_dist) {{
                min_dist = dist;
                material_id = f32(i + 1);
            }}
        }}

        // Environment sphere
        let env_dist = -sdf_sphere(p, 50.0);
        if (env_dist < min_dist) {{
            min_dist = env_dist;
            material_id = 999.0;
        }}

        return vec2<f32>(min_dist, material_id);
    }}

    fn get_material(material_id: f32) -> Material {{
        var mat: Material;
        
        if (material_id < 0.5) {{
            // Floor
            mat.albedo = vec3<f32>(0.65, 0.65, 0.6);
            mat.metallic = 0.0;
            mat.roughness = 0.85;
            mat.ao = 1.0;
            mat.emission = vec3<f32>(0.0);
        }} else if (material_id > 998.0) {{
            // Environment
            mat.albedo = vec3<f32>(0.02, 0.02, 0.03);
            mat.metallic = 0.0;
            mat.roughness = 1.0;
            mat.ao = 1.0;
            mat.emission = vec3<f32>(0.0);
        }} else {{
            // Instance material
            let instance_idx = u32(material_id - 1.0);
            if (instance_idx < uniforms.instance_count) {{
                let instance = instance_buffer.instances[instance_idx];
                mat.albedo = instance.color.xyz;
                mat.metallic = instance.color.w;
                mat.roughness = instance.material.x;
                mat.ao = instance.material.y;
                mat.emission = instance.color.xyz * instance.material.z;
            }} else {{
                mat.albedo = vec3<f32>(0.5);
                mat.metallic = 0.0;
                mat.roughness = 0.5;
                mat.ao = 1.0;
                mat.emission = vec3<f32>(0.0);
            }}
        }}
        
        return mat;
    }}

    fn calculate_normal(p: vec3<f32>) -> vec3<f32> {{
        let eps = vec2<f32>(EPSILON, 0.0);
        let d = scene_sdf(p).x;
        let dx = scene_sdf(p + eps.xyy).x - scene_sdf(p - eps.xyy).x;
        let dy = scene_sdf(p + eps.yxy).x - scene_sdf(p - eps.yxy).x;
        let dz = scene_sdf(p + eps.yyx).x - scene_sdf(p - eps.yyx).x;
        return normalize(vec3<f32>(dx, dy, dz));
    }}

    fn ray_march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {{
        var depth = MIN_DIST;
        
        for (var i = 0; i < MAX_MARCHING_STEPS; i++) {{
            let p = ray_origin + depth * ray_dir;
            let result = scene_sdf(p);
            let dist = result.x;
            
            if (dist < EPSILON) {{
                return vec3<f32>(depth, result.y, 1.0);
            }}
            
            depth += dist;
            
            if (depth >= MAX_DIST) {{
                break;
            }}
        }}
        
        return vec3<f32>(MAX_DIST, 0.0, 0.0);
    }}

    fn soft_shadow(ray_origin: vec3<f32>, ray_dir: vec3<f32>, mint: f32, maxt: f32, k: f32) -> f32 {{
        var res = 1.0;
        var t = mint;
        
        for (var i = 0; i < 64; i++) {{
            let p = ray_origin + t * ray_dir;
            let h = scene_sdf(p).x;
            
            if (h < SHADOW_EPSILON) {{
                return 0.0;
            }}
            
            res = min(res, k * h / t);
            t += h;
            
            if (t >= maxt) {{
                break;
            }}
        }}
        
        return res;
    }}

    fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {{
        var occlusion = 0.0;
        var scale = 1.0;
        
        for (var i = 0; i < AO_SAMPLES; i++) {{
            let hr = AO_STEP * f32(i + 1);
            let aopos = p + n * hr;
            let dd = scene_sdf(aopos).x;
            occlusion += (hr - dd) * scale;
            scale *= 0.75;
        }}
        
        return clamp(1.0 - 3.0 * occlusion, 0.0, 1.0);
    }}

    // GGX distribution
    fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {{
        let a = roughness * roughness;
        let a2 = a * a;
        let NdotH = max(dot(N, H), 0.0);
        let NdotH2 = NdotH * NdotH;
        
        let num = a2;
        let denom = (NdotH2 * (a2 - 1.0) + 1.0);
        let denom2 = denom * denom * 3.14159265359;
        
        return num / denom2;
    }}

    fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {{
        let r = (roughness + 1.0);
        let k = (r * r) / 8.0;
        return NdotV / (NdotV * (1.0 - k) + k);
    }}

    fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {{
        let NdotV = max(dot(N, V), 0.0);
        let NdotL = max(dot(N, L), 0.0);
        let ggx2 = geometry_schlick_ggx(NdotV, roughness);
        let ggx1 = geometry_schlick_ggx(NdotL, roughness);
        return ggx1 * ggx2;
    }}

    fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {{
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    }}

    fn calculate_lighting_tiled(hit_pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>, material: Material, frag_coord: vec2<f32>) -> vec3<f32> {{
        let ao = ambient_occlusion(hit_pos, normal);
        let ambient = vec3<f32>(0.03) * material.albedo * ao;
        
        var Lo = vec3<f32>(0.0);
        let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);
        
        let tile_idx = get_tile_index(frag_coord);
        let tile_data = tile_lights[tile_idx];
        
        for (var i: u32 = 0u; i < tile_data.light_count; i++) {{
            let light_idx = tile_data.light_indices[i];
            let light = light_buffer.lights[light_idx];
            
            if (light.intensity <= 0.0) {{
                continue;
            }}
            
            let light_vec = light.position - hit_pos;
            let distance = length(light_vec);
            
            if (distance < 0.001) {{
                continue;
            }}
            
            let L = normalize(light_vec);
            let H = normalize(view_dir + L);
            
            let shadow = soft_shadow(hit_pos + normal * SHADOW_EPSILON * 10.0, L, 0.02, distance, 32.0);
            
            if (shadow <= 0.0) {{
                continue;
            }}
            
            let attenuation = light.intensity / (
                light.attenuation_constant + 
                light.attenuation_linear * distance + 
                light.attenuation_quadratic * distance * distance
            );
            
            let radiance = light.color * attenuation * shadow;
            
            let NDF = distribution_ggx(normal, H, material.roughness);
            let G = geometry_smith(normal, view_dir, L, material.roughness);
            let F = fresnel_schlick(max(dot(H, view_dir), 0.0), F0);
            
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - material.metallic);
            
            let NdotL = max(dot(normal, L), 0.0);
            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(normal, view_dir), 0.0) * NdotL + 0.0001;
            let specular = numerator / denominator;
            
            Lo += (kD * material.albedo / 3.14159265359 + specular) * radiance * NdotL;
        }}
        
        Lo += material.emission;
        
        return ambient + Lo;
    }}

    fn calculate_lighting(hit_pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>, material: Material) -> vec3<f32> {{
        let ao = ambient_occlusion(hit_pos, normal);
        let ambient = vec3<f32>(0.03) * material.albedo * ao;
        
        var Lo = vec3<f32>(0.0);
        let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);
        
        let light_count = min(uniforms.light_count, {max_lights}u);
        for (var i: u32 = 0u; i < light_count; i++) {{
            let light = light_buffer.lights[i];
            
            if (light.intensity <= 0.0) {{
                continue;
            }}
            
            let light_vec = light.position - hit_pos;
            let distance = length(light_vec);
            
            if (distance < 0.001) {{
                continue;
            }}
            
            let L = normalize(light_vec);
            let H = normalize(view_dir + L);
            
            let shadow = soft_shadow(hit_pos + normal * SHADOW_EPSILON * 10.0, L, 0.02, distance, 32.0);
            
            if (shadow <= 0.0) {{
                continue;
            }}
            
            let attenuation = light.intensity / (
                light.attenuation_constant + 
                light.attenuation_linear * distance + 
                light.attenuation_quadratic * distance * distance
            );
            
            let radiance = light.color * attenuation * shadow;
            
            let NDF = distribution_ggx(normal, H, material.roughness);
            let G = geometry_smith(normal, view_dir, L, material.roughness);
            let F = fresnel_schlick(max(dot(H, view_dir), 0.0), F0);
            
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS) * (1.0 - material.metallic);
            
            let NdotL = max(dot(normal, L), 0.0);
            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(normal, view_dir), 0.0) * NdotL + 0.0001;
            let specular = numerator / denominator;
            
            Lo += (kD * material.albedo / 3.14159265359 + specular) * radiance * NdotL;
        }}
        
        Lo += material.emission;
        
        return ambient + Lo;
    }}

    fn get_tile_index(frag_coord: vec2<f32>) -> u32 {{
        let tile_x = u32(frag_coord.x / f32({lighting_tile_size}u));
        let tile_y = u32(frag_coord.y / f32({lighting_tile_size}u));
        
        let clamped_x = min(tile_x, compute_uniforms.tile_counts.x - 1u);
        let clamped_y = min(tile_y, compute_uniforms.tile_counts.y - 1u);
        
        return clamped_y * compute_uniforms.tile_counts.x + clamped_x;
    }}

    fn generate_ray_dir(vertex_pos: vec3<f32>) -> vec3<f32> {{
        let ndc = vec2<f32>(vertex_pos.x, vertex_pos.y);
        let clip_pos = vec4<f32>(ndc, -1.0, 1.0);
        
        let inv_view_proj = inverse_mat4(uniforms.view_proj);
        let world_pos = inv_view_proj * clip_pos;
        let ray_end = world_pos.xyz / world_pos.w;
        
        return normalize(ray_end - uniforms.camera_position);
    }}

    @vertex
    fn vs_main(vertex: VertexInput) -> VertexOutput {{
        var out: VertexOutput;
        out.clip_position = vec4<f32>(vertex.position.x, vertex.position.y, 0.0, 1.0);
        out.ray_dir = generate_ray_dir(vertex.position);
        out.uv = vertex.uv;
        return out;
    }}

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
        let ray_origin = uniforms.camera_position;
        let ray_dir = normalize(in.ray_dir);
        
        let march_result = ray_march(ray_origin, ray_dir);
        let t = march_result.x;
        let material_id = march_result.y;
        let hit = march_result.z > 0.5;
        
        if (!hit) {{
            let env_color = vec3<f32>(0.02, 0.02, 0.03);
            return vec4<f32>(env_color, 1.0);
        }}
        
        let hit_pos = ray_origin + t * ray_dir;
        let normal = calculate_normal(hit_pos);
        let view_dir = -ray_dir;
        
        let material = get_material(material_id);
        
        var color: vec3<f32>;
        if (compute_uniforms.tile_counts.x > 0u && compute_uniforms.tile_counts.y > 0u) {{
            let ndc = in.clip_position.xy / in.clip_position.w;
            let screen_coord = (ndc * 0.5 + 0.5) * vec2<f32>(compute_uniforms.screen_dims);
            color = calculate_lighting_tiled(hit_pos, normal, view_dir, material, screen_coord);
        }} else {{
            color = calculate_lighting(hit_pos, normal, view_dir, material);
        }}
        
        // ACES tone mapping
        let a = 2.51;
        let b = 0.03;
        let c = 2.43;
        let d = 0.59;
        let e = 0.14;
        let aces_color = (color * (a * color + b)) / (color * (c * color + d) + e);
        
        let final_color = pow(aces_color, vec3<f32>(1.0 / 2.2));
        
        return vec4<f32>(final_color, 1.0);
    }}
    "#, lighting_tile_size = lighting_tile_size, max_lights = max_lights, max_instances = max_instances)
}

fn generate_compute_shader(workgroup_size: u32, lighting_tile_size: u32, max_lights: usize) -> String {
    format!(r#"
    const LIGHTING_TILE_SIZE: u32 = {lighting_tile_size}u;

    struct ComputeUniforms {{
        screen_dims: vec2<u32>,
        tile_counts: vec2<u32>,
        time: f32,
        delta_time: f32,
    }}

    struct LightData {{
        position: vec3<f32>,
        intensity: f32,
        color: vec3<f32>,
        attenuation_constant: f32,
        attenuation_linear: f32,
        attenuation_quadratic: f32,
        _padding0: f32,
        _padding1: f32,
        _padding2: f32,
        _padding3: f32,
        _padding4: f32,
        _padding5: f32,
    }}

    struct LightBuffer {{
        lights: array<LightData, {max_lights}>,
    }}

    struct TileLightData {{
        light_indices: array<u32, {max_lights}>,
        light_count: u32,
    }}

    @group(0) @binding(0) var<uniform> compute_uniforms: ComputeUniforms;
    @group(0) @binding(1) var<storage, read_write> tile_lights: array<TileLightData>;
    @group(0) @binding(2) var<uniform> light_buffer: LightBuffer;
    @group(0) @binding(3) var<uniform> uniforms: Uniforms;

    struct Uniforms {{
        view_proj: mat4x4<f32>,
        camera_position: vec3<f32>,
        light_count: u32,
        time: f32,
        frame_index: u32,
        instance_count: u32,
    }}

    fn get_tile_frustum(tile_x: u32, tile_y: u32) -> array<vec4<f32>, 4> {{
        let tile_size = vec2<f32>(f32(LIGHTING_TILE_SIZE), f32(LIGHTING_TILE_SIZE));
        let screen_size = vec2<f32>(compute_uniforms.screen_dims);
        
        let min_x = (f32(tile_x) * tile_size.x) / screen_size.x * 2.0 - 1.0;
        let max_x = (f32(tile_x + 1u) * tile_size.x) / screen_size.x * 2.0 - 1.0;
        let min_y = 1.0 - (f32(tile_y + 1u) * tile_size.y) / screen_size.y * 2.0;
        let max_y = 1.0 - (f32(tile_y) * tile_size.y) / screen_size.y * 2.0;
        
        var planes: array<vec4<f32>, 4>;
        planes[0] = vec4<f32>(1.0, 0.0, 0.0, -min_x);
        planes[1] = vec4<f32>(-1.0, 0.0, 0.0, max_x);
        planes[2] = vec4<f32>(0.0, 1.0, 0.0, -min_y);
        planes[3] = vec4<f32>(0.0, -1.0, 0.0, max_y);
        
        return planes;
    }}

    fn light_affects_tile(light: LightData, tile_planes: array<vec4<f32>, 4>) -> bool {{
        if (light.intensity <= 0.0) {{
            return false;
        }}
        
        let threshold = 0.005;
        let a = light.attenuation_quadratic;
        let b = light.attenuation_linear;
        let c = light.attenuation_constant - light.intensity / threshold;
        
        var radius: f32;
        if (a > 0.0) {{
            let discriminant = b * b - 4.0 * a * c;
            if (discriminant >= 0.0) {{
                radius = (-b + sqrt(discriminant)) / (2.0 * a);
            }} else {{
                radius = 50.0;
            }}
        }} else if (b > 0.0) {{
            radius = -c / b;
        }} else {{
            radius = 50.0;
        }}
        
        radius = clamp(radius, 0.1, 150.0);
        
        let world_pos = vec4<f32>(light.position, 1.0);
        let clip_pos = uniforms.view_proj * world_pos;
        
        if (clip_pos.w <= 0.0) {{
            return false;
        }}
        
        let ndc = clip_pos.xyz / clip_pos.w;
        let screen_radius = radius / clip_pos.w;
        
        let tile_min_x = tile_planes[0].w;
        let tile_max_x = -tile_planes[1].w;
        let tile_min_y = tile_planes[2].w;
        let tile_max_y = -tile_planes[3].w;
        
        return ndc.x + screen_radius >= tile_min_x && 
            ndc.x - screen_radius <= tile_max_x &&
            ndc.y + screen_radius >= tile_min_y && 
            ndc.y - screen_radius <= tile_max_y;
    }}

    @compute @workgroup_size({workgroup_size}, {workgroup_size}, 1)
    fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
        let tile_x = global_id.x;
        let tile_y = global_id.y;
        
        if (tile_x >= compute_uniforms.tile_counts.x || tile_y >= compute_uniforms.tile_counts.y) {{
            return;
        }}
        
        let tile_index = tile_y * compute_uniforms.tile_counts.x + tile_x;
        var tile_data: TileLightData;
        tile_data.light_count = 0u;
        
        let tile_planes = get_tile_frustum(tile_x, tile_y);
        
        let max_lights = min(uniforms.light_count, {max_lights}u);
        for (var i = 0u; i < max_lights; i++) {{
            if (light_affects_tile(light_buffer.lights[i], tile_planes)) {{
                tile_data.light_indices[tile_data.light_count] = i;
                tile_data.light_count += 1u;
                
                if (tile_data.light_count >= {max_lights}u) {{
                    break;
                }}
            }}
        }}
        
        tile_lights[tile_index] = tile_data;
    }}
    "#, workgroup_size = workgroup_size, lighting_tile_size = lighting_tile_size, max_lights = max_lights)
}

// ============================
// === CORE IMPLEMENTATIONS ===
// ============================

impl UnifiedCore {
    pub fn new() -> Self {
        Self {
            transform: TransformSystem::new(),
            render: RenderSystem::new(),
            compute: ComputeSystem::new(),
            input: InputSystem::new(),
            timing: TimingSystem::new(),
            state_flags: StateFlags::RUNNING,
            stats_accumulator: StatsAccumulator::default(),
        }
    }

    #[inline(always)]
    pub fn update_and_render(&mut self, window_state: &mut WindowState) -> Result<(), wgpu::SurfaceError> {
        let current_time = self.timing.elapsed_seconds();
        let raw_delta = current_time - self.timing.last_frame_time;
        
        if raw_delta < FRAME_TIME {
            return Ok(());
        }
        
        self.timing.delta_time = raw_delta;
        self.timing.last_frame_time = current_time;
        self.timing.frame_count += 1;
        
        self.process_input();
        
        if !self.state_flags.contains(StateFlags::PAUSED) || self.transform.dirty_flags != 0 {
            self.update_systems(window_state);
        }
        
        if MINIMAL_LOGGING || LOGGING_ENABLED {
            self.stats_accumulator.frame_times[self.stats_accumulator.frame_index] = self.timing.delta_time;
            self.stats_accumulator.frame_index = (self.stats_accumulator.frame_index + 1) % 60;
            self.stats_accumulator.update_timer += self.timing.delta_time;
            
            if self.stats_accumulator.update_timer >= STATS_UPDATE_INTERVAL {
                self.update_stats();
            }
        }
        
        self.render_frame(window_state)
    }

    fn update_compute_uniforms(&self, window_state: &WindowState) {
        if self.compute.enabled && self.compute.compute_uniform_buffer.is_some() {
            let (tile_x, tile_y) = if self.state_flags.contains(StateFlags::USE_COMPUTE_LIGHTING) {
                (self.compute.tile_count_x, self.compute.tile_count_y)
            } else {
                (0, 0)
            };
            
            let compute_uniforms = ComputeUniforms {
                screen_dims: [DIMX, DIMY],
                tile_counts: [tile_x, tile_y],
                time: self.timing.elapsed_seconds(),
                delta_time: self.timing.delta_time,
                _padding: [0.0; 2],
            };
            
            window_state.queue.write_buffer(
                self.compute.compute_uniform_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&[compute_uniforms]),
            );
        }
    }

    #[inline(always)]
    fn process_input(&mut self) {
        let pressed = self.input.current_keys & !self.input.prev_keys;
        let released = self.input.prev_keys & !self.input.current_keys;
        
        if pressed.contains(InputFlags::P) {
            self.state_flags.toggle(StateFlags::PAUSED);
        }
        if pressed.contains(InputFlags::TAB) {
            let was_auto_rotating = self.state_flags.contains(StateFlags::CAMERA_AUTO_ROTATE);
            self.state_flags.toggle(StateFlags::CAMERA_AUTO_ROTATE);
            
            if was_auto_rotating && !self.state_flags.contains(StateFlags::CAMERA_AUTO_ROTATE) {
                self.transform.sync_rotation_from_target();
            }
        }
        if pressed.contains(InputFlags::DIGIT2) {
            self.state_flags.toggle(StateFlags::LIGHT_ANIMATION);
        }
        
        if !self.compute.enabled {
            self.state_flags.remove(StateFlags::USE_COMPUTE_LIGHTING);
        } else {
            if pressed.contains(InputFlags::C) {
                self.state_flags.toggle(StateFlags::USE_COMPUTE_LIGHTING);

                #[cfg(not(target_arch = "wasm32"))]
                println!("Compute lighting: {}", 
                    if self.state_flags.contains(StateFlags::USE_COMPUTE_LIGHTING) { 
                        "ENABLED (Tile-based)" 
                    } else { 
                        "DISABLED (Simple)" 
                    });
                
                #[cfg(target_arch = "wasm32")]
                web_sys::console::log_1(&format!("Compute lighting: {}", 
                    if self.state_flags.contains(StateFlags::USE_COMPUTE_LIGHTING) { 
                        "ENABLED (Tile-based)" 
                    } else { 
                        "DISABLED (Simple)" 
                    }).into());
            }
        }
        
        self.input.prev_keys = self.input.current_keys;
    }

    fn toggle_pointer_lock(&mut self) {
        if self.input.mouse_captured {
            self.release_pointer_lock();
        } else {
            self.transform.sync_rotation_from_target();
            let look_dir = (self.transform.camera_target - self.transform.camera_position).normalize();
            self.transform.camera_target = self.transform.camera_position + look_dir;
            
            self.input.mouse_captured = true;
            self.input.pointer_lock_frames = 5;
            self.input.mouse_delta = (0.0, 0.0);
            self.input.pointer_lock_transition = true;
            
            self.input.mouse_pos = (DIMX as f32 / 2.0, DIMY as f32 / 2.0);
            
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        if let Some(canvas) = document.get_element_by_id("canvas") {
                            let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
                            let _ = canvas.request_pointer_lock();
                        }
                    }
                }
            }
            
            #[cfg(not(target_arch = "wasm32"))]
            {
                println!("Pointer lock enabled - Rotation synced to ({:.1}°, {:.1}°)", 
                        self.transform.camera_rotation.y.to_degrees(),
                        self.transform.camera_rotation.x.to_degrees());
            }
        }
    }
    
    fn release_pointer_lock(&mut self) {
        self.input.mouse_captured = false;
        self.input.pointer_lock_transition = false;
        
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    document.exit_pointer_lock();
                }
            }
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("Pointer lock disabled");
        }
    }

    #[inline(always)]
    fn update_systems(&mut self, window_state: &WindowState) {
        if self.state_flags.contains(StateFlags::PAUSED) && 
        self.transform.dirty_flags == 0 && 
        !self.state_flags.contains(StateFlags::LIGHT_ANIMATION) {
            return;
        }

        if self.input.pointer_lock_frames > 0 {
            self.input.pointer_lock_frames -= 1;
            self.input.mouse_delta = (0.0, 0.0);
        }

        let has_input = self.input.current_keys.intersects(InputFlags::MOVEMENT) || 
                    (self.input.mouse_captured && self.input.mouse_delta != (0.0, 0.0));
        
        if has_input {
            self.transform.update_camera_movement(&self.input, self.timing.delta_time);
            self.input.mouse_delta = (0.0, 0.0);
        }
        
        if self.state_flags.contains(StateFlags::CAMERA_AUTO_ROTATE) {
            self.transform.camera_rotation.y += CAMERA_ROTATION_SPEED_Y * self.timing.delta_time;
            
            let sin_y = self.transform.camera_rotation.y.sin();
            let cos_y = self.transform.camera_rotation.y.cos();
            let radius = (self.transform.camera_position.x * self.transform.camera_position.x + 
                        self.transform.camera_position.z * self.transform.camera_position.z).sqrt().max(10.0);
            
            self.transform.camera_position.x = radius * sin_y;
            self.transform.camera_position.z = radius * cos_y;
            self.transform.camera_target = Vec3::new(0.0, 2.0, 0.0);
            self.transform.dirty_flags |= DIRTY_CAMERA;
        }
        
        if self.state_flags.contains(StateFlags::LIGHT_ANIMATION) {
            self.transform.animate_lights(self.timing.elapsed_seconds());
        }
        
        if self.transform.dirty_flags != 0 {
            self.update_gpu_resources(window_state);
        }
    }

    fn update_gpu_resources(&mut self, window_state: &WindowState) {
        if self.transform.dirty_flags == 0 {
            return;
        }
        
        if self.transform.dirty_flags & (DIRTY_CAMERA | DIRTY_LIGHTS) != 0 {
            self.transform.rebuild_matrices();
        }
        
        if let Some(buffer) = &self.render.uniform_buffer {
            let uniforms = Uniforms {
                view_proj: self.transform.view_proj_matrix.to_cols_array_2d(),
                camera_position: self.transform.camera_position.to_array(),
                light_count: self.transform.light_count,
                time: self.timing.elapsed_seconds(),
                frame_index: self.timing.frame_count,
                instance_count: self.render.instances.len() as u32,
                _padding: [0.0; 1],
            };
            
            window_state.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[uniforms]));
        }
        
        if self.transform.dirty_flags & DIRTY_LIGHTS != 0 {
            if let Some(buffer) = &self.render.light_buffer {
                let light_buffer = LightBuffer { lights: self.transform.lights };
                window_state.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[light_buffer]));
            }
        }
        
        if self.transform.dirty_flags & (DIRTY_INSTANCES | DIRTY_CAMERA) != 0 && 
        !self.render.instances.is_empty() {
            
            if self.state_flags.contains(StateFlags::FRUSTUM_CULLING) {
                self.frustum_cull_instances_simd();
            } else if self.render.visible_instances.len() != self.render.instances.len() {
                self.render.visible_instances = (0..self.render.instances.len() as u32).collect();
            }
            
            if !self.render.visible_instances.is_empty() && self.render.instance_buffer.is_some() {
                let mut visible_data = Vec::with_capacity(self.render.visible_instances.len());
                unsafe {
                    visible_data.set_len(self.render.visible_instances.len());
                    for (i, &idx) in self.render.visible_instances.iter().enumerate() {
                        *visible_data.get_unchecked_mut(i) = *self.render.instances.get_unchecked(idx as usize);
                    }
                }
                
                window_state.queue.write_buffer(
                    self.render.instance_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&visible_data),
                );
            }
        }
        
        self.transform.dirty_flags = 0;
    }

    fn frustum_cull_instances_simd(&mut self) {
        self.render.visible_instances.clear();
        
        let frustum = self.transform.calculate_frustum();
        let planes = &frustum.planes;
        
        let mut plane_data = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 6];
        for (i, plane) in planes.iter().enumerate() {
            plane_data[i] = (plane.x, plane.y, plane.z, plane.w);
        }
        
        const RADIUS: f32 = 5.0; // Larger radius for SDF objects
        const NEG_RADIUS: f32 = -RADIUS;
        
        for (idx, instance) in self.render.instances.iter().enumerate() {
            let pos_x = instance.model_matrix[3][0];
            let pos_y = instance.model_matrix[3][1];
            let pos_z = instance.model_matrix[3][2];
            
            let mut inside = true;
            
            for &(px, py, pz, pw) in &plane_data {
                let distance = px * pos_x + py * pos_y + pz * pos_z + pw;
                if distance < NEG_RADIUS {
                    inside = false;
                    break;
                }
            }
            
            if inside {
                self.render.visible_instances.push(idx as u32);
            }
        }
    }

    #[inline(always)]
    fn render_frame(&self, window_state: &mut WindowState) -> Result<(), wgpu::SurfaceError> {
        let output = window_state.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(window_state.config.format.add_srgb_suffix()),
            ..Default::default()
        });
        
        let mut encoder = window_state.device.create_command_encoder(&Default::default());
        
        self.update_compute_uniforms(window_state);
        
        if self.compute.enabled && 
        self.state_flags.contains(StateFlags::USE_COMPUTE_LIGHTING) &&
        self.compute.compute_pipeline.is_some() {
            
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Lighting Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(self.compute.compute_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, self.compute.compute_bind_group.as_ref().unwrap(), &[]);
            
            let workgroups_x = (self.compute.tile_count_x + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
            let workgroups_y = (self.compute.tile_count_y + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

            if LOGGING_ENABLED && self.timing.frame_count % LOGGING_TIMESTEP == 0 {
                #[cfg(not(target_arch = "wasm32"))]
                println!("Compute pass: {}x{} workgroups for {}x{} tiles", 
                    workgroups_x, workgroups_y, self.compute.tile_count_x, self.compute.tile_count_y);
            }
        }
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &window_state.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let pipeline = match &self.render.render_pipeline {
                Some(p) => p,
                None => return Ok(()),
            };
            
            render_pass.set_pipeline(pipeline);
            
            if let Some(bind_group) = &self.render.bind_group {
                render_pass.set_bind_group(0, bind_group, &[]);
            }
            
            if let (Some(vb), Some(ib)) = (&self.render.vertex_buffer, &self.render.index_buffer) {
                render_pass.set_vertex_buffer(0, vb.slice(..));
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.render.num_indices, 0, 0..1);
            }
        }

        window_state.queue.submit([encoder.finish()]);
        window_state.window.pre_present_notify();
        output.present();
        
        Ok(())
    }

    fn update_stats(&mut self) {
        self.stats_accumulator.frame_times[self.stats_accumulator.frame_index] = self.timing.delta_time;
        self.stats_accumulator.frame_index = (self.stats_accumulator.frame_index + 1) % 60;
        self.stats_accumulator.update_timer += self.timing.delta_time;
        
        if self.stats_accumulator.update_timer >= STATS_UPDATE_INTERVAL {
            self.stats_accumulator.update_timer = 0.0;
            
            let sum = self.stats_accumulator.frame_times.iter().sum::<f32>();
            let avg_frame_time = sum / 60.0;
            let fps = 1.0 / avg_frame_time;
            
            #[cfg(not(target_arch = "wasm32"))]
            if MINIMAL_LOGGING {
                println!("FPS: {:.1}, Frame: {:.2}ms, SDFs: {}, Lights: {}", 
                    fps, avg_frame_time * 1000.0,
                    self.render.instances.len(),
                    self.transform.light_count);
            }
            
            #[cfg(target_arch = "wasm32")]
            if MINIMAL_LOGGING {
                use std::fmt::Write;
                self.stats_accumulator.stats_buffer.clear();
                let _ = write!(&mut self.stats_accumulator.stats_buffer, 
                    "<div>FPS: {:.1}</div><div>Frame: {:.2}ms</div><div>SDFs: {}</div><div>Lights: {}</div>",
                    fps, avg_frame_time * 1000.0,
                    self.render.instances.len(),
                    self.transform.light_count
                );
                
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        if let Some(stats_element) = document.get_element_by_id("stats") {
                            stats_element.set_inner_html(&self.stats_accumulator.stats_buffer);
                        }
                    }
                }
            }
        }
    }

    pub fn handle_key_input(&mut self, physical_key: &PhysicalKey, pressed: bool) {
        if LOGGING_ENABLED {
            #[cfg(not(target_arch = "wasm32"))]
            {
                println!("Physical Key {:?} detected", physical_key);
            }
            #[cfg(target_arch = "wasm32")]
            {
                web_sys::console::log_1(&format!("Physical Key {:?} detected", physical_key).into());
            }
        }

        let flag = match physical_key {
            PhysicalKey::Code(WinitKeyCode::KeyW) => Some(InputFlags::W),
            PhysicalKey::Code(WinitKeyCode::KeyA) => Some(InputFlags::A),
            PhysicalKey::Code(WinitKeyCode::KeyS) => Some(InputFlags::S),
            PhysicalKey::Code(WinitKeyCode::KeyD) => Some(InputFlags::D),
            PhysicalKey::Code(WinitKeyCode::KeyQ) => Some(InputFlags::Q),
            PhysicalKey::Code(WinitKeyCode::KeyE) => Some(InputFlags::E),
            PhysicalKey::Code(WinitKeyCode::KeyR) => Some(InputFlags::R),
            PhysicalKey::Code(WinitKeyCode::KeyL) => Some(InputFlags::L),
            PhysicalKey::Code(WinitKeyCode::KeyP) => Some(InputFlags::P),
            PhysicalKey::Code(WinitKeyCode::KeyI) => Some(InputFlags::I),
            PhysicalKey::Code(WinitKeyCode::KeyC) => Some(InputFlags::C),
            PhysicalKey::Code(WinitKeyCode::KeyF) => Some(InputFlags::F),
            PhysicalKey::Code(WinitKeyCode::Space) => Some(InputFlags::SPACE),
            PhysicalKey::Code(WinitKeyCode::Tab) => Some(InputFlags::TAB),
            PhysicalKey::Code(WinitKeyCode::Escape) => Some(InputFlags::ESC),
            PhysicalKey::Code(WinitKeyCode::Digit1) => Some(InputFlags::DIGIT1),
            PhysicalKey::Code(WinitKeyCode::Digit2) => Some(InputFlags::DIGIT2),
            PhysicalKey::Code(WinitKeyCode::ArrowUp) => Some(InputFlags::ARROW_UP),
            PhysicalKey::Code(WinitKeyCode::ArrowDown) => Some(InputFlags::ARROW_DOWN),
            PhysicalKey::Code(WinitKeyCode::ArrowLeft) => Some(InputFlags::ARROW_LEFT),
            PhysicalKey::Code(WinitKeyCode::ArrowRight) => Some(InputFlags::ARROW_RIGHT),
            _ => None,
        };

        if let Some(flag) = flag {
            if pressed {
                self.input.current_keys.insert(flag);
                
                match flag {
                    InputFlags::L => {
                        self.transform.cycle_primary_light_color();
                        
                        if LOGGING_ENABLED {
                            #[cfg(not(target_arch = "wasm32"))]
                            println!("Cycling light color");
                        }
                    },
                    InputFlags::R => self.reset_camera(),
                    InputFlags::ESC => {
                        if self.input.mouse_captured {
                            self.input.mouse_captured = false;
                            self.release_pointer_lock();
                        } else {
                            self.state_flags.insert(StateFlags::SHOULD_EXIT);
                        }
                    },
                    InputFlags::DIGIT1 => {
                        self.add_sdf_instance();

                        if LOGGING_ENABLED {
                            #[cfg(not(target_arch = "wasm32"))]
                            println!("Key '1' pressed - adding SDF instance");
                        }
                    },
                    InputFlags::F => {
                        self.toggle_pointer_lock();
                    },
                    InputFlags::ARROW_UP => self.transform.adjust_light_intensity(LIGHT_INTENSITY_STEP),
                    InputFlags::ARROW_DOWN => self.transform.adjust_light_intensity(-LIGHT_INTENSITY_STEP),
                    _ => {}
                }
            } else {
                self.input.current_keys.remove(flag);
            }
        }
    }

    fn reset_camera(&mut self) {
        self.transform.camera_position = Vec3::new(25.0, 5.0, 25.0);
        self.transform.camera_target = Vec3::new(0.0, 2.0, 0.0);
        self.transform.camera_fov = 45.0;
        
        self.transform.sync_rotation_from_target();
    }

    fn add_sdf_instance(&mut self) {
        if self.render.instances.len() >= MAX_INSTANCES {
            if LOGGING_ENABLED {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    println!("Max SDF instances reached: {}/{}", self.render.instances.len(), MAX_INSTANCES);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    web_sys::console::log_1(&"Max SDF instances reached".into());
                }
            }
            return;
        }

        let colors = [
            [1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0],
            [1.0, 0.5, 0.0], [0.8, 0.2, 1.0], [1.0, 1.0, 0.2],
        ];
        
        let count = self.render.instances.len() as f32;
        let angle = count * 0.7;
        let radius = 8.0 + (count * 0.5);
        let height = 2.0 + (count % 3.0) as f32 * 2.5;
        
        let position = Vec3::new(
            angle.cos() * radius,
            height,
            angle.sin() * radius
        );
        
        let color_idx = self.render.instances.len() % colors.len();
        let color = colors[color_idx];
        
        // Vary SDF type
        let sdf_type = (self.render.instances.len() % 4) as f32;
        
        let (sdf_params, scale) = match sdf_type as u32 {
            0 => ([1.5, 0.0, 0.0, 0.0], Vec3::ONE), // Sphere
            1 => ([1.2, 1.5, 1.0, 0.0], Vec3::ONE), // Box
            2 => ([1.0, 1.2, 0.8, 0.2], Vec3::ONE), // Rounded box
            3 => ([2.0, 0.5, 0.0, 0.0], Vec3::ONE), // Torus
            _ => ([1.0, 0.0, 0.0, 0.0], Vec3::ONE),
        };
        
        self.render.instances.push(InstanceData {
            model_matrix: Mat4::from_scale_rotation_translation(
                scale,
                glam::Quat::from_rotation_y(angle * 0.3),
                position
            ).to_cols_array_2d(),
            color: [color[0], color[1], color[2], 0.5], // Semi-metallic
            material: [0.3, 1.0, 0.0, sdf_type], // roughness, ao, emission, type
            sdf_params,
        });
        
        self.transform.dirty_flags |= DIRTY_INSTANCES;
        
        if LOGGING_ENABLED {
            #[cfg(not(target_arch = "wasm32"))]
            {
                println!("Added SDF instance {}: type={}, pos={:?}", 
                        self.render.instances.len() - 1, sdf_type, position);
            }
            #[cfg(target_arch = "wasm32")]
            {
                web_sys::console::log_1(&format!("Added SDF instance {}: type={}", 
                                self.render.instances.len() - 1, sdf_type).into());
            }
        }
    }

    pub fn should_exit(&self) -> bool {
        self.state_flags.contains(StateFlags::SHOULD_EXIT)
    }
}

// ======================================
// === TRANSFORM SYSTEM IMPLEMENTATION ===
// ======================================

impl TransformSystem {
    pub fn new() -> Self {
        let mut lights = [LightData {
            position: [0.0, 0.0, 0.0],
            intensity: 0.0,
            color: [1.0, 1.0, 1.0],
            attenuation_constant: 1.0,
            attenuation_linear: 0.09,
            attenuation_quadratic: 0.032,
            _padding: [0.0; 6],
        }; MAX_LIGHTS];

        lights[0] = LightData {
            position: [5.0, 18.0, 5.0],
            intensity: 2.0,
            color: [1.0, 0.9, 0.7],
            attenuation_constant: 1.0,
            attenuation_linear: 0.022,
            attenuation_quadratic: 0.0019,
            _padding: [0.0; 6],
        };

        lights[1] = LightData {
            position: [-8.0, 12.0, 8.0],
            intensity: 1.5,
            color: [0.3, 0.5, 1.0],
            attenuation_constant: 1.0,
            attenuation_linear: 0.09,
            attenuation_quadratic: 0.032,
            _padding: [0.0; 6],
        };

        let camera_position = Vec3::new(25.0, 5.0, 25.0);
        let camera_target = Vec3::new(0.0, 2.0, 0.0);
        let camera_fov: f32 = 45.0;
        let camera_aspect = DIMX as f32 / DIMY as f32;

        let proj_matrix = Mat4::perspective_rh(camera_fov.to_radians(), camera_aspect, 0.1, 100.0);
        let view_matrix = Mat4::look_at_rh(camera_position, camera_target, Vec3::Y);
        let view_proj_matrix = proj_matrix * view_matrix;

        let mut transform = Self {
            view_matrix,
            proj_matrix,
            view_proj_matrix,
            camera_position,
            camera_target,
            camera_rotation: Vec3::ZERO,
            camera_fov,
            camera_aspect,
            lights,
            light_count: 2,
            animation_time: 0.0,
            dirty_flags: DIRTY_CAMERA | DIRTY_LIGHTS,
        };
        
        transform.sync_rotation_from_target();
        
        transform
    }

    pub fn sync_rotation_from_target(&mut self) {
        let look_dir = (self.camera_target - self.camera_position).normalize();
        self.camera_rotation.x = look_dir.y.asin();
        self.camera_rotation.y = look_dir.x.atan2(-look_dir.z);
        self.dirty_flags |= DIRTY_CAMERA;
    }

    #[inline(always)]
    pub fn update_camera_movement(&mut self, input: &InputSystem, delta_time: f32) {
        let speed = CAMERA_MOVE_SPEED * delta_time;
        
        if input.mouse_captured && (input.mouse_delta.0 != 0.0 || input.mouse_delta.1 != 0.0) {
            self.camera_rotation.y += input.mouse_delta.0 * 0.003;
            self.camera_rotation.x = (self.camera_rotation.x - input.mouse_delta.1 * 0.003).clamp(-1.5, 1.5);
            self.dirty_flags |= DIRTY_CAMERA;
        }
        
        let (sin_yaw, cos_yaw) = self.camera_rotation.y.sin_cos();
        let (sin_pitch, cos_pitch) = self.camera_rotation.x.sin_cos();
        
        let look_direction = Vec3::new(
            sin_yaw * cos_pitch,
            sin_pitch,
            -cos_yaw * cos_pitch
        );
        
        self.camera_target = self.camera_position + look_direction;
        
        if input.current_keys.intersects(InputFlags::MOVEMENT) {
            let forward = Vec3::new(sin_yaw, 0.0, -cos_yaw);
            let right = Vec3::new(cos_yaw, 0.0, sin_yaw);
            
            let keys = input.current_keys;
            let fwd = (keys.contains(InputFlags::W) as i32 - keys.contains(InputFlags::S) as i32) as f32;
            let strafe = (keys.contains(InputFlags::D) as i32 - keys.contains(InputFlags::A) as i32) as f32;
            let vert = (keys.contains(InputFlags::SPACE) as i32 - keys.contains(InputFlags::Q) as i32) as f32;
            
            if fwd != 0.0 || strafe != 0.0 || vert != 0.0 {
                let movement = (forward * fwd + right * strafe + Vec3::Y * vert).normalize_or_zero() * speed;
                self.camera_position += movement;
                self.camera_target = self.camera_position + look_direction;
                self.dirty_flags |= DIRTY_CAMERA;
            }
        }
    }

    #[inline(always)]
    pub fn rebuild_matrices(&mut self) {
        self.proj_matrix = Mat4::perspective_rh(self.camera_fov.to_radians(), self.camera_aspect, 0.1, 100.0);
        self.view_matrix = Mat4::look_at_rh(self.camera_position, self.camera_target, Vec3::Y);
        self.view_proj_matrix = self.proj_matrix * self.view_matrix;
    }

    #[inline(always)]
    pub fn animate_lights(&mut self, time: f32) {
        if self.light_count == 0 {
            return;
        }
        
        let base_time = time * LIGHT_ANIMATION_SPEED;
        let (sin_base, cos_base) = base_time.sin_cos();
        let sin_half = (base_time * 0.5).sin();
        
        let count = self.light_count.min(MAX_LIGHTS as u32) as usize;
        let inv_count = 1.0 / self.light_count.max(1) as f32;
        
        if count <= 4 {
            for i in 0..count {
                let phase = i as f32 * std::f32::consts::TAU * inv_count;
                let (sin_phase, cos_phase) = phase.sin_cos();
                
                self.lights[i].position = [
                    LIGHT_ANIMATION_RADIUS * (cos_base * cos_phase - sin_base * sin_phase),
                    LIGHT_ANIMATION_HEIGHT + 2.0 * sin_half,
                    LIGHT_ANIMATION_RADIUS * (sin_base * cos_phase + cos_base * sin_phase)
                ];
            }
        } else {
            let phase_step = std::f32::consts::TAU * inv_count;
            for i in 0..count {
                let phase = i as f32 * phase_step;
                let (sin_phase, cos_phase) = phase.sin_cos();
                
                self.lights[i].position = [
                    LIGHT_ANIMATION_RADIUS * (cos_base * cos_phase - sin_base * sin_phase),
                    LIGHT_ANIMATION_HEIGHT + 2.0 * sin_half,
                    LIGHT_ANIMATION_RADIUS * (sin_base * cos_phase + cos_base * sin_phase)
                ];
            }
        }
        
        self.dirty_flags |= DIRTY_LIGHTS;
    }

    pub fn cycle_primary_light_color(&mut self) {
        if self.light_count > 0 {
            let current = &self.lights[0].color;
            self.lights[0].color = match *current {
                [1.0, 0.9, 0.7] => [0.8, 0.9, 1.0],
                [0.8, 0.9, 1.0] => [1.0, 0.2, 0.2],
                [1.0, 0.2, 0.2] => [0.2, 1.0, 0.2],
                [0.2, 1.0, 0.2] => [0.2, 0.2, 1.0],
                _ => [1.0, 0.9, 0.7],
            };
            self.dirty_flags |= DIRTY_LIGHTS;
        }
    }

    pub fn adjust_light_intensity(&mut self, delta: f32) {
        if self.light_count > 0 {
            self.lights[0].intensity = (self.lights[0].intensity + delta).clamp(0.1, 10.0);
            self.dirty_flags |= DIRTY_LIGHTS;
        }
    }

    pub fn update_aspect_ratio(&mut self, aspect: f32) {
        self.camera_aspect = aspect;
        self.dirty_flags |= DIRTY_CAMERA;
    }

    pub fn calculate_frustum(&self) -> Frustum {
        Frustum::from_matrix(&self.view_proj_matrix)
    }
}

pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(view_proj: &Mat4) -> Self {
        let m = view_proj.to_cols_array_2d();
        
        let planes = [
            Vec4::new(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]),
            Vec4::new(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]),
            Vec4::new(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]),
            Vec4::new(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]),
            Vec4::new(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]),
            Vec4::new(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]),
        ];
        
        Self { planes }
    }

    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let distance = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if distance < -radius {
                return false;
            }
        }
        true
    }
}

// ======================================
// === HELPER IMPLEMENTATIONS ===
// ======================================

impl RenderSystem {
    pub fn new() -> Self {
        Self {
            uniform_buffer: None,
            light_buffer: None,
            instance_buffer: None,
            render_pipeline: None,
            bind_group: None,
            bind_group_layout: None,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            instances: Vec::new(),
            visible_instances: Vec::new(),
        }
    }
}

impl ComputeSystem {
    pub fn new() -> Self {
        let tile_count_x = (DIMX + LIGHTING_TILE_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
        let tile_count_y = (DIMY + LIGHTING_TILE_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
        
        Self {
            enabled: false,
            compute_pipeline: None,
            compute_bind_group: None,
            compute_uniform_buffer: None,
            lighting_buffer: None,
            tile_count_x,
            tile_count_y,
        }
    }
}

impl InputSystem {
    pub fn new() -> Self {
        Self {
            current_keys: InputFlags::empty(),
            prev_keys: InputFlags::empty(),
            mouse_pos: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
            mouse_captured: false,
            pointer_lock_frames: 0,
            centering_camera: false,
            pointer_lock_transition: false,
        }
    }
}

impl TimingSystem {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            start_time_ms: Self::now_ms(),
            last_frame_time: 0.0,
            delta_time: 0.0,
            accumulator: 0.0,
            frame_count: 0,
        }
    }

    pub fn elapsed_seconds(&self) -> f32 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f32()
        }

        #[cfg(target_arch = "wasm32")]
        {
            let now_ms = Self::now_ms();
            ((now_ms - self.start_time_ms) / 1000.0) as f32
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn now_ms() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now())
    }
}

// ======================================
// === WINDOW STATE ===
// ======================================

pub struct WindowState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl WindowState {
    pub async fn new(window: Arc<Window>) -> WindowState {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let size = winit::dpi::PhysicalSize::new(DIMX, DIMY);
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::BROWSER_WEBGPU,
                    ..Default::default()
                });
                let limits = wgpu::Limits::downlevel_webgl2_defaults();
            } else {
                let size = window.inner_size();
                let instance = wgpu::Instance::default();
                let limits = wgpu::Limits::default();
            }
        }

        let surface = instance.create_surface(window.clone()).expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: Default::default(),
                    trace: Default::default(),
                }
            )
            .await
            .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if cfg!(target_arch = "wasm32") {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::Immediate
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.add_srgb_suffix()],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let (depth_texture, depth_view) = Self::create_depth_texture(&device, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            depth_texture,
            depth_view,
        }
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (depth_texture, depth_view)
    }

    pub fn initialize_render_data(&self, core: &mut UnifiedCore) {
        let uniform_size = std::mem::size_of::<Uniforms>() as u64;
        
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_buffer_size = std::mem::size_of::<LightBuffer>() as u64;
        let light_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Buffer"),
            size: light_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<InstanceData>() * MAX_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_count = core.compute.tile_count_x * core.compute.tile_count_y;
        let tile_data_size = std::mem::size_of::<u32>() * (8 + 1);
        let dummy_lighting_buffer_size = (tile_count as usize * tile_data_size) as u64;
        
        let dummy_lighting_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Lighting Tile Buffer"),
            size: dummy_lighting_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let dummy_compute_uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Compute Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ComputeUniforms {
                screen_dims: [DIMX, DIMY],
                tile_counts: [0, 0],
                time: 0.0,
                delta_time: 0.0,
                _padding: [0.0; 2],
            }]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let initial_uniforms = Uniforms {
            view_proj: core.transform.view_proj_matrix.to_cols_array_2d(),
            camera_position: core.transform.camera_position.to_array(),
            light_count: core.transform.light_count,
            time: 0.0,
            frame_index: 0,
            instance_count: 0,
            _padding: [0.0; 1],
        };
        
        self.queue.write_buffer(
            &uniform_buffer,
            0,
            bytemuck::cast_slice(&[initial_uniforms]),
        );
        
        let initial_lights = LightBuffer {
            lights: core.transform.lights,
        };
        
        self.queue.write_buffer(
            &light_buffer,
            0,
            bytemuck::cast_slice(&[initial_lights]),
        );

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let dummy_buffers = (dummy_lighting_buffer, dummy_compute_uniform_buffer);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_buffers.0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_buffers.1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
            label: Some("bind_group"),
        });

        let (vertices, indices, instances) = Self::create_test_scene();
        
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let render_shader_source = generate_render_shader(LIGHTING_TILE_SIZE, MAX_LIGHTS, MAX_INSTANCES);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(render_shader_source.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Ray March Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        core.render.uniform_buffer = Some(uniform_buffer);
        core.render.light_buffer = Some(light_buffer);
        core.render.instance_buffer = Some(instance_buffer);
        core.render.render_pipeline = Some(render_pipeline);
        core.render.bind_group = Some(bind_group);
        core.render.bind_group_layout = Some(bind_group_layout);
        core.render.vertex_buffer = Some(vertex_buffer);
        core.render.index_buffer = Some(index_buffer);
        core.render.num_indices = indices.len() as u32;
        core.render.instances = instances;
        
        core.render.visible_instances = (0..core.render.instances.len() as u32).collect();
        
        core.transform.dirty_flags = DIRTY_CAMERA | DIRTY_LIGHTS | DIRTY_INSTANCES;
        
        self.initialize_compute_resources(core);
    }

    fn initialize_compute_resources(&self, core: &mut UnifiedCore) {
        let compute_uniforms = ComputeUniforms {
            screen_dims: [DIMX, DIMY],
            tile_counts: [core.compute.tile_count_x, core.compute.tile_count_y],
            time: 0.0,
            delta_time: 0.0,
            _padding: [0.0; 2],
        };
        
        let compute_uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Uniform Buffer"),
            contents: bytemuck::cast_slice(&[compute_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tile_count = core.compute.tile_count_x * core.compute.tile_count_y;
        let tile_data_size = std::mem::size_of::<u32>() * (8 + 1);
        let lighting_buffer_size = (tile_count as usize * tile_data_size) as u64;
        
        let lighting_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lighting Tile Buffer"),
            size: lighting_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

        let compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: core.render.light_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: core.render.uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
            label: Some("compute_bind_group"),
        });

        let compute_shader_source = generate_compute_shader(
            COMPUTE_WORKGROUP_SIZE,
            LIGHTING_TILE_SIZE,
            MAX_LIGHTS
        );
        let compute_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(compute_shader_source.into()),
        });

        let compute_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Lighting Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        core.compute.enabled = true;
        core.compute.compute_pipeline = Some(compute_pipeline);
        core.compute.compute_bind_group = Some(compute_bind_group);
        core.compute.compute_uniform_buffer = Some(compute_uniform_buffer.clone());
        core.compute.lighting_buffer = Some(lighting_buffer.clone());
        
        core.state_flags.insert(StateFlags::USE_COMPUTE_LIGHTING);

        if let (Some(bind_group_layout), Some(uniform_buffer), Some(light_buffer), Some(instance_buffer)) = 
            (&core.render.bind_group_layout, &core.render.uniform_buffer, &core.render.light_buffer, &core.render.instance_buffer) {
            
            let new_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lighting_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: compute_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: instance_buffer.as_entire_binding(),
                    },
                ],
                label: Some("bind_group_with_compute"),
            });
            
            core.render.bind_group = Some(new_bind_group);
        }
    }

    fn create_test_scene() -> (Vec<Vertex>, Vec<u16>, Vec<InstanceData>) {
        let vertices = vec![
            Vertex { position: [-1.0, -1.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0] },
            Vertex { position: [ 1.0, -1.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
            Vertex { position: [ 1.0,  1.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0] },
            Vertex { position: [-1.0,  1.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] },
        ];
        
        let indices = vec![0, 1, 2, 2, 3, 0];
        
        let mut instances = vec![];
        
        // Hero sphere - metallic chrome
        instances.push(InstanceData {
            model_matrix: Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::IDENTITY,
                Vec3::new(0.0, 2.0, 0.0)
            ).to_cols_array_2d(),
            color: [0.95, 0.93, 0.88, 1.0], // Chrome
            material: [0.05, 1.0, 0.0, 0.0], // Very smooth, SDF type 0 = sphere
            sdf_params: [1.5, 0.0, 0.0, 0.0], // Radius 1.5
        });
        
        // Left cube - blue plastic
        instances.push(InstanceData {
            model_matrix: Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::from_rotation_y(0.5),
                Vec3::new(-3.5, 1.5, -2.0)
            ).to_cols_array_2d(),
            color: [0.1, 0.3, 0.8, 0.0], // Blue, non-metallic
            material: [0.3, 1.0, 0.0, 2.0], // Plastic roughness, SDF type 2 = rounded box
            sdf_params: [1.2, 1.5, 1.2, 0.1], // Size and rounding
        });
        
        // Right box - emissive
        instances.push(InstanceData {
            model_matrix: Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::from_rotation_y(-0.3),
                Vec3::new(3.0, 1.0, -1.5)
            ).to_cols_array_2d(),
            color: [1.0, 0.8, 0.4, 0.0], // Orange
            material: [0.2, 1.0, 2.0, 1.0], // Emissive, SDF type 1 = box
            sdf_params: [0.8, 1.0, 0.8, 0.0], // Box size
        });
        
        if LOGGING_ENABLED {
            #[cfg(not(target_arch = "wasm32"))]
            {
                println!("Created SDF scene with {} instances", instances.len());
            }
            #[cfg(target_arch = "wasm32")]
            {
                web_sys::console::log_1(&format!("Created SDF scene with {} instances", 
                            instances.len()).into());
            }
        }
        
        (vertices, indices, instances)
    }
}

// ======================================
// === UNIFIED APPLICATION ===
// ======================================

pub struct UnifiedApp {
    pub core: UnifiedCore,
    pub window_state: Option<WindowState>,
    pub window: Option<Arc<Window>>,
    #[cfg(target_arch = "wasm32")]
    pub state_initializing: bool,
}

impl Default for UnifiedApp {
    fn default() -> Self {
        Self {
            core: UnifiedCore::new(),
            window_state: None,
            window: None,
            #[cfg(target_arch = "wasm32")]
            state_initializing: false,
        }
    }
}

impl UnifiedApp {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(target_arch = "wasm32")]
    fn initialize_stats_display(&mut self) {
        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                if let Some(app_container) = document.get_element_by_id("app") {
                    let app_style = app_container.dyn_ref::<web_sys::HtmlElement>().unwrap().style();
                    app_style.set_property("position", "relative").unwrap();
                    
                    let stats_div = document.create_element("div").unwrap();
                    stats_div.set_id("stats");
                    
                    let style = stats_div.dyn_ref::<web_sys::HtmlElement>().unwrap().style();
                    style.set_property("position", "absolute").unwrap();
                    style.set_property("top", "10px").unwrap();
                    style.set_property("left", "10px").unwrap();
                    style.set_property("z-index", "1000").unwrap();
                    style.set_property("background", "rgba(0,0,0,0.8)").unwrap();
                    style.set_property("color", "white").unwrap();
                    style.set_property("padding", "10px").unwrap();
                    style.set_property("font-family", "monospace").unwrap();
                    style.set_property("font-size", "12px").unwrap();
                    style.set_property("border-radius", "4px").unwrap();
                    
                    app_container.append_child(&stats_div).unwrap();
                }
            }
        }
    }

    pub fn update(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window_state) = &mut self.window_state {
            if let Err(e) = self.core.update_and_render(window_state) {
                match e {
                    wgpu::SurfaceError::Lost => {
                        window_state.surface.configure(&window_state.device, &window_state.config);
                    }
                    wgpu::SurfaceError::OutOfMemory => {
                        self.core.state_flags.insert(StateFlags::SHOULD_EXIT);
                    }
                    _ => {
                        #[cfg(target_arch = "wasm32")]
                        web_sys::console::error_1(&format!("Render error: {:?}", e).into());
                        #[cfg(not(target_arch = "wasm32"))]
                        eprintln!("Render error: {:?}", e);
                    }
                }
            }
        }

        if self.core.should_exit() {
            _event_loop.exit();
        }

        self.request_next_frame();
    }

    fn request_next_frame(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler for UnifiedApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("LayerW v0.4.0 - SDF Instanced")
                        .with_inner_size(winit::dpi::PhysicalSize::new(DIMX, DIMY))
                )
                .unwrap(),
        );

        window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(DIMX, DIMY)));
        window.set_max_inner_size(Some(winit::dpi::PhysicalSize::new(DIMX, DIMY)));
        window.set_resizable(false);
        
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;

            let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(DIMX, DIMY));

            if let Some(canvas) = window.canvas() {
                let web_window = web_sys::window().unwrap();
                let document = web_window.document().unwrap();
                
                let container = document.get_element_by_id("app")
                    .unwrap_or_else(|| document.body().unwrap().into());
                
                canvas.set_width(DIMX.into());
                canvas.set_height(DIMY.into());
                
                let style = canvas.style();
                style.set_property("width", &format!("{}px", DIMX)).unwrap();
                style.set_property("height", &format!("{}px", DIMY)).unwrap();
                style.set_property("max-width", &format!("{}px", DIMX)).unwrap();
                style.set_property("max-height", &format!("{}px", DIMY)).unwrap();
                
                container.append_child(&web_sys::Element::from(canvas))
                    .expect("Couldn't append canvas to document");
            }
            
            self.window = Some(window.clone());
            self.state_initializing = true;
            
            let window_clone = window.clone();
            let app_ptr = self as *mut UnifiedApp;
            wasm_bindgen_futures::spawn_local(async move {
                let state = WindowState::new(window_clone).await;
                unsafe {
                    let app = &mut *app_ptr;
                    state.initialize_render_data(&mut app.core);
                    app.window_state = Some(state);
                    app.state_initializing = false;
                    app.core.state_flags.insert(StateFlags::RUNNING);
                }
            });
            
            if MINIMAL_LOGGING || LOGGING_ENABLED {
                self.initialize_stats_display();
            }
            
            window.request_redraw();
            return;
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(WindowState::new(window.clone()));
            state.initialize_render_data(&mut self.core);
            self.window_state = Some(state);
            self.window = Some(window.clone());
            self.core.state_flags.insert(StateFlags::RUNNING);
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let window = match &self.window {
            Some(window) => window,
            None => return,
        };
        
        if window.id() != id {
            return;
        }

        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                self.core.handle_key_input(&event.physical_key, event.state == winit::event::ElementState::Pressed);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let flag = match button {
                    WinitMouseButton::Left => InputFlags::MOUSE_LEFT,
                    WinitMouseButton::Right => InputFlags::MOUSE_RIGHT,
                    WinitMouseButton::Middle => InputFlags::MOUSE_MIDDLE,
                    _ => return,
                };
                
                if state == winit::event::ElementState::Pressed {
                    self.core.input.current_keys.insert(flag);
                } else {
                    self.core.input.current_keys.remove(flag);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.core.input.mouse_captured {
                    if self.core.input.pointer_lock_frames > 0 {
                        self.core.input.mouse_pos = (DIMX as f32 / 2.0, DIMY as f32 / 2.0);
                        self.core.input.mouse_delta = (0.0, 0.0);
                        return;
                    }
                    
                    if self.core.input.pointer_lock_transition {
                        self.core.input.pointer_lock_transition = false;
                        self.core.input.mouse_pos = (position.x as f32, position.y as f32);
                        self.core.input.mouse_delta = (0.0, 0.0);
                        return;
                    }
                    
                    let old_pos = self.core.input.mouse_pos;
                    self.core.input.mouse_pos = (position.x as f32, position.y as f32);
                    
                    let raw_delta = (
                        position.x as f32 - old_pos.0,
                        position.y as f32 - old_pos.1
                    );
                    
                    const MAX_DELTA: f32 = 100.0;
                    self.core.input.mouse_delta = (
                        raw_delta.0.clamp(-MAX_DELTA, MAX_DELTA),
                        raw_delta.1.clamp(-MAX_DELTA, MAX_DELTA)
                    );
                } else {
                    self.core.input.mouse_pos = (position.x as f32, position.y as f32);
                    self.core.input.mouse_delta = (0.0, 0.0);
                }
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(window_state) = &mut self.window_state {
                    window_state.config.width = physical_size.width;
                    window_state.config.height = physical_size.height;
                    window_state.surface.configure(&window_state.device, &window_state.config);
                    
                    let aspect_ratio = physical_size.width as f32 / physical_size.height as f32;
                    self.core.transform.update_aspect_ratio(aspect_ratio);
                    
                    let (depth_texture, depth_view) = WindowState::create_depth_texture(&window_state.device, &window_state.config);
                    window_state.depth_texture = depth_texture;
                    window_state.depth_view = depth_view;
                }
            }
            WindowEvent::CloseRequested => {
                self.core.state_flags.insert(StateFlags::SHOULD_EXIT);
            }
            WindowEvent::RedrawRequested => {
                self.update(event_loop);
            }
            _ => {}
        }
    }
}

// ======================================
// === MAIN ENTRY POINT ===
// ======================================

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    if MINIMAL_LOGGING || LOGGING_ENABLED {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
                web_sys::console::log_1(&"Started LayerW v0.4.0 - SDF Instanced".into());
            } else {
                env_logger::init();
                log::info!("Started LayerW v0.4.0 - SDF Instanced");
            }
        }
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = UnifiedApp::new();
    event_loop.run_app(&mut app).unwrap();
}