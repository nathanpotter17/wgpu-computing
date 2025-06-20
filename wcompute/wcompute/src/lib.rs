use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures;

const ENHANCED_3D_SHADER: &str = r#"
struct Vertex {
    position: vec3<f32>,
    _padding1: f32,
    normal: vec3<f32>,
    _padding2: f32,
    uv: vec2<f32>,
    _padding3: array<f32, 6>,
}

struct Material {
    diffuse: vec3<f32>,
    _padding1: f32,
    specular: vec3<f32>, 
    shininess: f32,
}

struct PointLight {
    position: vec3<f32>,
    _padding1: f32,
    color: vec3<f32>,
    intensity: f32,
    attenuation: f32,
    _padding2: array<f32, 7>,
}

struct SceneUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    screen_width: u32,
    screen_height: u32,
    total_triangles: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> materials: array<Material>;
@group(0) @binding(3) var<storage, read> lights: array<PointLight>;
@group(0) @binding(4) var<uniform> scene: SceneUniforms;
@group(0) @binding(5) var<storage, read_write> framebuffer: array<u32>;
@group(0) @binding(6) var<storage, read_write> depth_buffer: array<f32>;

fn blinn_phong(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light: PointLight,
    material: Material
) -> vec3<f32> {
    let light_dir = normalize(light.position - world_pos);
    let distance = length(light.position - world_pos);
    
    let attenuation = light.intensity / (1.0 + light.attenuation * distance * distance);
    let diffuse = max(0.0, dot(normal, light_dir));
    
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(0.0, dot(normal, half_dir)), material.shininess);
    
    return (material.diffuse * diffuse + material.specular * specular) * light.color * attenuation;
}

// Enhanced compute shader that can handle larger workgroups
@compute @workgroup_size(64, 1, 1)  // Using 1D workgroup for maximum flexibility
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = global_id.x;
    let total_pixels = scene.screen_width * scene.screen_height;
    
    if (pixel_index >= total_pixels) {
        return;
    }
    
    let pixel_x = pixel_index % scene.screen_width;
    let pixel_y = pixel_index / scene.screen_width;
    let pixel_pos = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);
    
    var closest_depth = 1000.0;
    var final_color = vec3<f32>(0.1, 0.1, 0.2); // Dark blue background
    
    // Process all triangles for this pixel
    let triangle_count = scene.total_triangles;
    for (var tri_idx = 0u; tri_idx < triangle_count; tri_idx++) {
        let i0 = indices[tri_idx * 3u];
        let i1 = indices[tri_idx * 3u + 1u];
        let i2 = indices[tri_idx * 3u + 2u];
        
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        
        // Transform vertices to clip space
        let world_pos0 = vec4<f32>(v0.position, 1.0);
        let world_pos1 = vec4<f32>(v1.position, 1.0);
        let world_pos2 = vec4<f32>(v2.position, 1.0);
        
        let view_pos0 = scene.view_matrix * world_pos0;
        let view_pos1 = scene.view_matrix * world_pos1;
        let view_pos2 = scene.view_matrix * world_pos2;
        
        let clip_pos0 = scene.proj_matrix * view_pos0;
        let clip_pos1 = scene.proj_matrix * view_pos1;
        let clip_pos2 = scene.proj_matrix * view_pos2;
        
        // Skip if all vertices are behind camera
        if (clip_pos0.w <= 0.0 && clip_pos1.w <= 0.0 && clip_pos2.w <= 0.0) {
            continue;
        }
        
        // Convert to screen space
        let ndc0 = clip_pos0.xyz / clip_pos0.w;
        let ndc1 = clip_pos1.xyz / clip_pos1.w;
        let ndc2 = clip_pos2.xyz / clip_pos2.w;
        
        let screen0 = vec2<f32>(
            (ndc0.x + 1.0) * 0.5 * f32(scene.screen_width),
            (1.0 - ndc0.y) * 0.5 * f32(scene.screen_height)
        );
        let screen1 = vec2<f32>(
            (ndc1.x + 1.0) * 0.5 * f32(scene.screen_width),
            (1.0 - ndc1.y) * 0.5 * f32(scene.screen_height)
        );
        let screen2 = vec2<f32>(
            (ndc2.x + 1.0) * 0.5 * f32(scene.screen_width),
            (1.0 - ndc2.y) * 0.5 * f32(scene.screen_height)
        );
        
        // Barycentric coordinates
        let v0v1 = screen1 - screen0;
        let v0v2 = screen2 - screen0;
        let v0p = pixel_pos - screen0;
        
        let denom = v0v1.x * v0v2.y - v0v2.x * v0v1.y;
        if (abs(denom) < 0.001) { continue; }
        
        let v = (v0p.x * v0v2.y - v0v2.x * v0p.y) / denom;
        let w = (v0v1.x * v0p.y - v0p.x * v0v1.y) / denom;
        let u = 1.0 - v - w;
        
        // Check if point is inside triangle
        if (u < 0.0 || v < 0.0 || w < 0.0) {
            continue;
        }
        
        // Interpolate depth
        let depth = u * ndc0.z + v * ndc1.z + w * ndc2.z;
        
        // Depth test
        if (depth > closest_depth) {
            continue;
        }
        
        closest_depth = depth;
        
        // Interpolate world position and normal
        let world_pos = u * world_pos0.xyz + v * world_pos1.xyz + w * world_pos2.xyz;
        let normal = normalize(u * v0.normal + v * v1.normal + w * v2.normal);
        
        // Lighting calculation
        if (arrayLength(&materials) > 0u && arrayLength(&lights) > 0u) {
            let material = materials[0];
            let view_dir = normalize(scene.camera_pos - world_pos);
            
            var color = material.diffuse * 0.1; // Ambient
            
            for (var light_idx = 0u; light_idx < arrayLength(&lights); light_idx++) {
                color += blinn_phong(world_pos, normal, view_dir, lights[light_idx], material);
            }
            
            final_color = color;
        }
    }
    
    // Store depth
    depth_buffer[pixel_index] = closest_depth;
    
    // Gamma correction and pack to RGBA8
    final_color = pow(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    
    let r = u32(final_color.r * 255.0);
    let g = u32(final_color.g * 255.0);
    let b = u32(final_color.b * 255.0);
    framebuffer[pixel_index] = 0xFF000000u | (b << 16u) | (g << 8u) | r;
}
"#;

// Updated struct with total triangles
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneUniforms {
    pub view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub total_triangles: u32,
    pub _padding: u32,
}

// Keep the same vertex, material, and light structs
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlinnPhongVertex {
    pub position: [f32; 3],
    pub _padding1: f32,
    pub normal: [f32; 3], 
    pub _padding2: f32,
    pub uv: [f32; 2],
    pub _padding3: [f32; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlinnPhongMaterial {
    pub diffuse: [f32; 3],
    pub _padding1: f32,
    pub specular: [f32; 3],
    pub shininess: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointLight {
    pub position: [f32; 3],
    pub _padding1: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub attenuation: f32,
    pub _padding2: [f32; 7],
}

pub struct EnhancedNRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    material_buffer: Option<wgpu::Buffer>,
    light_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    framebuffer: Option<wgpu::Buffer>,
    depth_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,
    
    screen_width: u32,
    screen_height: u32,
}

impl EnhancedNRenderer {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find adapter");
        
        // Enhanced device limits for full WebGPU compute capabilities
        let mut limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_defaults()
        } else {
            wgpu::Limits::default()
        };
        
        // Override key limits for compute shaders
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_size_y = 256;
        limits.max_compute_workgroup_size_z = 64;
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_buffer_size = 1024 * 1024 * 1024; // 1GB for large scenes
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024; // 128MB per buffer
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Enhanced 3D Renderer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;
        
        Self::create_renderer(device, queue)
    }
    
    fn create_renderer(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, String> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Enhanced 3D Renderer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Enhanced 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(ENHANCED_3D_SHADER.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Enhanced 3D Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Enhanced 3D Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Uniforms"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            vertex_buffer: None,
            index_buffer: None,
            material_buffer: None,
            light_buffer: None,
            uniform_buffer,
            framebuffer: None,
            depth_buffer: None,
            staging_buffer: None,
            screen_width: 0,
            screen_height: 0,
        })
    }
    
    pub async fn render_frame(
        &mut self,
        vertices: &[BlinnPhongVertex],
        indices: &[u32],
        materials: &[BlinnPhongMaterial],
        lights: &[PointLight],
        mut scene_uniforms: SceneUniforms,
    ) -> Result<Vec<u8>, String> {
        let width = scene_uniforms.screen_width;
        let height = scene_uniforms.screen_height;
        let pixel_count = (width * height) as usize;
        
        // Update total triangles in scene uniforms
        scene_uniforms.total_triangles = (indices.len() / 3) as u32;
        
        if self.screen_width != width || self.screen_height != height {
            self.resize_buffers(width, height)?;
        }
        
        self.update_vertex_buffer(vertices)?;
        self.update_index_buffer(indices)?;
        self.update_material_buffer(materials)?;
        self.update_light_buffer(lights)?;
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&scene_uniforms));
        
        self.clear_buffers().await?;
        
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Enhanced 3D Render Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.vertex_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.index_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.material_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.framebuffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.depth_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });
        
        // Enhanced dispatch: Use 1D workgroups for maximum flexibility
        // With workgroup size 64, we can handle up to 65535 * 64 = ~4.2M pixels per dispatch
        let workgroups = (pixel_count as u32 + 63) / 64;  // Round up to nearest 64
        
        // Handle very large framebuffers with multiple dispatches if needed
        let max_workgroups = 65535;
        let dispatches_needed = (workgroups + max_workgroups - 1) / max_workgroups;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Enhanced 3D Render Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Enhanced 3D Render Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // For most cases, we'll only need one dispatch
            if dispatches_needed == 1 {
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            } else {
                // For extremely large framebuffers, we'd need multiple dispatches
                // This is rare but possible with 8K+ resolutions
                for dispatch in 0..dispatches_needed {
                    let remaining_workgroups = workgroups - (dispatch * max_workgroups);
                    let current_workgroups = remaining_workgroups.min(max_workgroups);
                    compute_pass.dispatch_workgroups(current_workgroups, 1, 1);
                }
            }
        }
        
        let framebuffer_size = (pixel_count * 4) as u64;
        encoder.copy_buffer_to_buffer(
            self.framebuffer.as_ref().unwrap(),
            0,
            self.staging_buffer.as_ref().unwrap(),
            0,
            framebuffer_size,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        self.read_framebuffer().await
    }
    
    // Rest of the implementation remains the same...
    fn resize_buffers(&mut self, width: u32, height: u32) -> Result<(), String> {
        let pixel_count = (width * height) as usize;
        let framebuffer_size = (pixel_count * 4) as u64;
        let depth_size = (pixel_count * 4) as u64;
        
        self.framebuffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Framebuffer"),
            size: framebuffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        self.depth_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Buffer"),
            size: depth_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: framebuffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        
        self.screen_width = width;
        self.screen_height = height;
        Ok(())
    }
    
    fn update_vertex_buffer(&mut self, vertices: &[BlinnPhongVertex]) -> Result<(), String> {
        if self.vertex_buffer.is_none() || 
           self.vertex_buffer.as_ref().unwrap().size() < (vertices.len() * std::mem::size_of::<BlinnPhongVertex>()) as u64 {
            self.vertex_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        } else {
            self.queue.write_buffer(self.vertex_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(vertices));
        }
        Ok(())
    }
    
    fn update_index_buffer(&mut self, indices: &[u32]) -> Result<(), String> {
        if self.index_buffer.is_none() || 
           self.index_buffer.as_ref().unwrap().size() < (indices.len() * 4) as u64 {
            self.index_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        } else {
            self.queue.write_buffer(self.index_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(indices));
        }
        Ok(())
    }
    
    fn update_material_buffer(&mut self, materials: &[BlinnPhongMaterial]) -> Result<(), String> {
        if self.material_buffer.is_none() || 
           self.material_buffer.as_ref().unwrap().size() < (materials.len() * std::mem::size_of::<BlinnPhongMaterial>()) as u64 {
            self.material_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material Buffer"),
                contents: bytemuck::cast_slice(materials),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        } else {
            self.queue.write_buffer(self.material_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(materials));
        }
        Ok(())
    }
    
    fn update_light_buffer(&mut self, lights: &[PointLight]) -> Result<(), String> {
        if self.light_buffer.is_none() || 
           self.light_buffer.as_ref().unwrap().size() < (lights.len() * std::mem::size_of::<PointLight>()) as u64 {
            self.light_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Light Buffer"),
                contents: bytemuck::cast_slice(lights),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        } else {
            self.queue.write_buffer(self.light_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(lights));
        }
        Ok(())
    }
    
    async fn clear_buffers(&self) -> Result<(), String> {
        let pixel_count = self.screen_width * self.screen_height;
        
        // Clear depth buffer to far plane (1000.0)
        self.queue.write_buffer(
            self.depth_buffer.as_ref().unwrap(),
            0,
            &vec![1000.0f32; pixel_count as usize].as_slice().iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        
        // Clear framebuffer to background color
        self.queue.write_buffer(
            self.framebuffer.as_ref().unwrap(),
            0,
            &vec![0xFF331A0Du32; pixel_count as usize].as_slice().iter()
                .flat_map(|&u| u.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        
        Ok(())
    }
    
    async fn read_framebuffer(&self) -> Result<Vec<u8>, String> {
        let buffer_slice = self.staging_buffer.as_ref().unwrap().slice(..);
        
        #[cfg(target_arch = "wasm32")]
        {
            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    match result {
                        Ok(()) => { resolve.call0(&wasm_bindgen::JsValue::NULL).unwrap(); },
                        Err(e) => { reject.call1(&wasm_bindgen::JsValue::NULL, &format!("{:?}", e).into()).unwrap(); },
                    }
                });
            });
            
            let _ = self.device.poll(wgpu::MaintainBase::Poll);
            wasm_bindgen_futures::JsFuture::from(promise).await.map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device.poll(wgpu::MaintainBase::Wait);
            receiver.recv().map_err(|_| "Channel receive failed".to_string())?
                .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
        }
        
        let data = buffer_slice.get_mapped_range();
        let rgba_data = data.to_vec();
        drop(data);
        self.staging_buffer.as_ref().unwrap().unmap();
        
        Ok(rgba_data)
    }
    
    // Enhanced demo scene with more complex geometry to showcase performance
    pub fn create_complex_demo_scene() -> (Vec<BlinnPhongVertex>, Vec<u32>, Vec<BlinnPhongMaterial>, Vec<PointLight>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        // Create multiple cubes in a grid pattern to stress test the renderer
        let grid_size = 5;
        let spacing = 3.0;
        let mut vertex_offset = 0;
        
        for x in 0..grid_size {
            for y in 0..grid_size {
                for z in 0..grid_size {
                    let offset_x = (x as f32 - grid_size as f32 / 2.0) * spacing;
                    let offset_y = (y as f32 - grid_size as f32 / 2.0) * spacing;
                    let offset_z = (z as f32 - grid_size as f32 / 2.0) * spacing;
                    
                    // Create cube vertices with offset
                    let cube_vertices = vec![
                        // Front face
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x,  0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x,  0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        
                        // Back face
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x,  0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x,  0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        
                        // Left face
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x,  0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x,  0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        
                        // Right face
                        BlinnPhongVertex { position: [0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [0.5 + offset_x,  0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [0.5 + offset_x,  0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        
                        // Top face
                        BlinnPhongVertex { position: [-0.5 + offset_x, 0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x, 0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, 0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, 0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        
                        // Bottom face
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, -0.5 + offset_y, -0.5 + offset_z], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [ 0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
                        BlinnPhongVertex { position: [-0.5 + offset_x, -0.5 + offset_y,  0.5 + offset_z], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
                    ];
                    
                    vertices.extend(cube_vertices);
                    
                    // Add indices for this cube
                    let cube_indices = vec![
                        // Front face
                        0, 1, 2,   2, 3, 0,
                        // Back face  
                        4, 5, 6,   6, 7, 4,
                        // Left face
                        8, 9, 10,  10, 11, 8,
                        // Right face
                        12, 13, 14, 14, 15, 12,
                        // Top face
                        16, 17, 18, 18, 19, 16,
                        // Bottom face
                        20, 21, 22, 22, 23, 20,
                    ];
                    
                    for &index in &cube_indices {
                        indices.push(vertex_offset + index);
                    }
                    
                    vertex_offset += 24; // 24 vertices per cube
                }
            }
        }
        
        // Create multiple materials with different colors
        let materials = vec![
            BlinnPhongMaterial {
                diffuse: [0.8, 0.3, 0.2],
                _padding1: 0.0,
                specular: [0.8, 0.8, 0.8],
                shininess: 32.0,
            },
            BlinnPhongMaterial {
                diffuse: [0.2, 0.8, 0.3],
                _padding1: 0.0,
                specular: [0.9, 0.9, 0.9],
                shininess: 64.0,
            },
            BlinnPhongMaterial {
                diffuse: [0.3, 0.2, 0.8],
                _padding1: 0.0,
                specular: [0.7, 0.7, 0.9],
                shininess: 16.0,
            }
        ];
        
        // Multiple lights for better illumination of the complex scene
        let lights = vec![
            PointLight {
                position: [10.0, 10.0, 10.0],
                _padding1: 0.0,
                color: [1.0, 0.9, 0.8],
                intensity: 3.0,
                attenuation: 0.01,
                _padding2: [0.0; 7],
            },
            PointLight {
                position: [-10.0, 5.0, -5.0],
                _padding1: 0.0,
                color: [0.8, 0.9, 1.0],
                intensity: 2.0,
                attenuation: 0.015,
                _padding2: [0.0; 7],
            },
            PointLight {
                position: [0.0, -8.0, 8.0],
                _padding1: 0.0,
                color: [1.0, 1.0, 0.9],
                intensity: 1.5,
                attenuation: 0.02,
                _padding2: [0.0; 7],
            }
        ];
        
        (vertices, indices, materials, lights)
    }
    
    // Original simple demo scene
    pub fn create_demo_cube_scene() -> (Vec<BlinnPhongVertex>, Vec<u32>, Vec<BlinnPhongMaterial>, Vec<PointLight>) {
        let vertices = vec![
            // Front face
            BlinnPhongVertex { position: [-1.0, -1.0,  1.0], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, -1.0,  1.0], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0,  1.0,  1.0], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0,  1.0,  1.0], _padding1: 0.0, normal: [0.0, 0.0, 1.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            
            // Back face
            BlinnPhongVertex { position: [-1.0, -1.0, -1.0], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0,  1.0, -1.0], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0,  1.0, -1.0], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, -1.0, -1.0], _padding1: 0.0, normal: [0.0, 0.0, -1.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            
            // Left face
            BlinnPhongVertex { position: [-1.0, -1.0, -1.0], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0, -1.0,  1.0], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0,  1.0,  1.0], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0,  1.0, -1.0], _padding1: 0.0, normal: [-1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            
            // Right face
            BlinnPhongVertex { position: [1.0, -1.0, -1.0], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [1.0,  1.0, -1.0], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [1.0,  1.0,  1.0], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [1.0, -1.0,  1.0], _padding1: 0.0, normal: [1.0, 0.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            
            // Top face
            BlinnPhongVertex { position: [-1.0, 1.0, -1.0], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0, 1.0,  1.0], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, 1.0,  1.0], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, 1.0, -1.0], _padding1: 0.0, normal: [0.0, 1.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            
            // Bottom face
            BlinnPhongVertex { position: [-1.0, -1.0, -1.0], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [1.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, -1.0, -1.0], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [0.0, 1.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [ 1.0, -1.0,  1.0], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [0.0, 0.0], _padding3: [0.0; 6] },
            BlinnPhongVertex { position: [-1.0, -1.0,  1.0], _padding1: 0.0, normal: [0.0, -1.0, 0.0], _padding2: 0.0, uv: [1.0, 0.0], _padding3: [0.0; 6] },
        ];
        
        let indices = vec![
            // Front face
            0, 1, 2,   2, 3, 0,
            // Back face  
            4, 5, 6,   6, 7, 4,
            // Left face
            8, 9, 10,  10, 11, 8,
            // Right face
            12, 13, 14, 14, 15, 12,
            // Top face
            16, 17, 18, 18, 19, 16,
            // Bottom face
            20, 21, 22, 22, 23, 20,
        ];
        
        let materials = vec![
            BlinnPhongMaterial {
                diffuse: [0.8, 0.3, 0.2],
                _padding1: 0.0,
                specular: [0.8, 0.8, 0.8],
                shininess: 32.0,
            }
        ];
        
        let lights = vec![
            PointLight {
                position: [5.0, 5.0, 5.0],
                _padding1: 0.0,
                color: [1.0, 1.0, 1.0],
                intensity: 2.0,
                attenuation: 0.02,
                _padding2: [0.0; 7],
            }
        ];
        
        (vertices, indices, materials, lights)
    }
}

// Matrix utility functions - same as before
pub fn create_view_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize_vec3(subtract_vec3(target, eye));
    let s = normalize_vec3(cross_product(f, up));
    let u = cross_product(s, f);
    
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot_product(s, eye), -dot_product(u, eye), dot_product(f, eye), 1.0],
    ]
}

pub fn create_perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov * 0.5).tan();
    let range_inv = 1.0 / (near - far);
    
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (near + far) * range_inv, -1.0],
        [0.0, 0.0, 2.0 * near * far * range_inv, 0.0],
    ]
}

fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 0.0] }
}

fn subtract_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_product(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot_product(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Enhanced WASM interface with full WebGPU capabilities
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct EnhancedNWRenderer {
    inner: EnhancedNRenderer,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EnhancedNWRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> js_sys::Promise {
        console_error_panic_hook::set_once();
        wasm_bindgen_futures::future_to_promise(async move {
            match EnhancedNRenderer::new().await {
                Ok(renderer) => Ok(JsValue::from(EnhancedNWRenderer { inner: renderer })),
                Err(e) => Err(JsValue::from_str(&e)),
            }
        })
    }
    
    #[wasm_bindgen]
    pub fn render_demo_cube(&mut self, width: u32, height: u32, time: f32) -> js_sys::Promise {
        let (vertices, indices, materials, lights) = EnhancedNRenderer::create_demo_cube_scene();
        
        let camera_radius = 6.0;
        let camera_x = camera_radius * (time * 0.3).cos();
        let camera_z = camera_radius * (time * 0.3).sin();
        let camera_y = 2.0 + (time * 0.2).sin() * 1.0;
        
        let view_matrix = create_view_matrix(
            [camera_x, camera_y, camera_z],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        
        let proj_matrix = create_perspective_matrix(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        
        let scene_uniforms = SceneUniforms {
            view_matrix,
            proj_matrix,
            camera_pos: [camera_x, camera_y, camera_z],
            time,
            screen_width: width,
            screen_height: height,
            total_triangles: 0, // Will be set by render_frame
            _padding: 0,
        };
        
        let mut renderer = EnhancedNRenderer {
            device: self.inner.device.clone(),
            queue: self.inner.queue.clone(),
            pipeline: self.inner.pipeline.clone(),
            bind_group_layout: self.inner.bind_group_layout.clone(),
            vertex_buffer: None,
            index_buffer: None,
            material_buffer: None,
            light_buffer: None,
            uniform_buffer: self.inner.uniform_buffer.clone(),
            framebuffer: None,
            depth_buffer: None,
            staging_buffer: None,
            screen_width: 0,
            screen_height: 0,
        };
        
        wasm_bindgen_futures::future_to_promise(async move {
            match renderer.render_frame(&vertices, &indices, &materials, &lights, scene_uniforms).await {
                Ok(rgba_data) => {
                    let js_array = js_sys::Uint8Array::new_with_length(rgba_data.len() as u32);
                    js_array.copy_from(&rgba_data);
                    Ok(JsValue::from(js_array))
                }
                Err(e) => Err(JsValue::from_str(&e)),
            }
        })
    }
    
    #[wasm_bindgen]
    pub fn render_complex_scene(&mut self, width: u32, height: u32, time: f32) -> js_sys::Promise {
        let (vertices, indices, materials, lights) = EnhancedNRenderer::create_complex_demo_scene();
        
        // More dynamic camera for complex scene
        let camera_radius = 20.0;
        let camera_x = camera_radius * (time * 0.1).cos();
        let camera_z = camera_radius * (time * 0.1).sin();
        let camera_y = 8.0 + (time * 0.15).sin() * 3.0;
        
        let view_matrix = create_view_matrix(
            [camera_x, camera_y, camera_z],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        
        let proj_matrix = create_perspective_matrix(
            60.0f32.to_radians(), // Wider FOV for complex scene
            width as f32 / height as f32,
            0.1,
            200.0, // Longer far plane
        );
        
        let scene_uniforms = SceneUniforms {
            view_matrix,
            proj_matrix,
            camera_pos: [camera_x, camera_y, camera_z],
            time,
            screen_width: width,
            screen_height: height,
            total_triangles: 0, // Will be set by render_frame
            _padding: 0,
        };
        
        let mut renderer = EnhancedNRenderer {
            device: self.inner.device.clone(),
            queue: self.inner.queue.clone(),
            pipeline: self.inner.pipeline.clone(),
            bind_group_layout: self.inner.bind_group_layout.clone(),
            vertex_buffer: None,
            index_buffer: None,
            material_buffer: None,
            light_buffer: None,
            uniform_buffer: self.inner.uniform_buffer.clone(),
            framebuffer: None,
            depth_buffer: None,
            staging_buffer: None,
            screen_width: 0,
            screen_height: 0,
        };
        
        wasm_bindgen_futures::future_to_promise(async move {
            match renderer.render_frame(&vertices, &indices, &materials, &lights, scene_uniforms).await {
                Ok(rgba_data) => {
                    let js_array = js_sys::Uint8Array::new_with_length(rgba_data.len() as u32);
                    js_array.copy_from(&rgba_data);
                    Ok(JsValue::from(js_array))
                }
                Err(e) => Err(JsValue::from_str(&e)),
            }
        })
    }
    
    #[wasm_bindgen]
    pub fn get_device_limits(&self) -> js_sys::Object {
        let obj = js_sys::Object::new();
        
        // Note: These are the limits we're requesting, actual device limits may vary
        js_sys::Reflect::set(&obj, &"maxComputeWorkgroupsPerDimension".into(), &65535.into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxComputeWorkgroupSizeX".into(), &256.into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxComputeWorkgroupSizeY".into(), &256.into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxComputeWorkgroupSizeZ".into(), &64.into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxComputeInvocationsPerWorkgroup".into(), &256.into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxBufferSize".into(), &(1024_u64 * 1024 * 1024).into()).unwrap();
        js_sys::Reflect::set(&obj, &"maxStorageBufferBindingSize".into(), &(128_u64 * 1024 * 1024).into()).unwrap();
        
        obj
    }
}

// Performance testing utilities
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct PerformanceMetrics {
    triangle_count: u32,
    vertex_count: u32,
    pixel_count: u32,
    workgroup_count: u32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl PerformanceMetrics {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, triangles: u32, vertices: u32) -> PerformanceMetrics {
        let pixel_count = width * height;
        let workgroup_count = (pixel_count + 63) / 64; // Round up for workgroup size 64
        
        PerformanceMetrics {
            triangle_count: triangles,
            vertex_count: vertices,
            pixel_count,
            workgroup_count,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn triangle_count(&self) -> u32 { 
        self.triangle_count 
    }
    
    #[wasm_bindgen(getter)]
    pub fn vertex_count(&self) -> u32 { 
        self.vertex_count 
    }
    
    #[wasm_bindgen(getter)]
    pub fn pixel_count(&self) -> u32 { 
        self.pixel_count 
    }
    
    #[wasm_bindgen(getter)]
    pub fn workgroup_count(&self) -> u32 { 
        self.workgroup_count 
    }
    
    #[wasm_bindgen]
    pub fn theoretical_max_resolution_for_workgroups(&self) -> u32 {
        // With 65535 max workgroups and workgroup size 64
        let max_pixels = 65535 * 64;
        (max_pixels as f32).sqrt() as u32
    }
}