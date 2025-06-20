use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

const DIMX: u32 = 1080;
const DIMY: u32 = 720;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> State {
        // Configure instance based on platform
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

        let adapter_info = adapter.get_info();
        
        let adapter = if adapter_info.device_type == wgpu::DeviceType::IntegratedGpu {
            if let Ok(discrete_adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                })
                .await
            {
                let discrete_info = discrete_adapter.get_info();
                if discrete_info.device_type == wgpu::DeviceType::DiscreteGpu {
                    println!("Found better discrete GPU: {} ({:?})", discrete_info.name, discrete_info.device_type);

                    discrete_adapter
                } else {
                    adapter
                }
            } else {
                adapter
            }
        } else {
            adapter
        };

        // Get final adapter information
        let adapter_info = adapter.get_info();

        #[cfg(not(target_arch = "wasm32"))]
        println!("Selected GPU: {} ({:?})", adapter_info.name, adapter_info.device_type);

        #[cfg(target_arch = "wasm32")]
        {
            // Log the selected on web by looking through the navigator.
            use wasm_bindgen::prelude::*;
    
            let js_code = r#"
                if (navigator.gpu) {
                    console.log("WebGPU is supported");
                    console.log("Hardware concurrency: " + navigator.hardwareConcurrency);

                    navigator.gpu.requestAdapter().then((adapter) => {
                        if (adapter) {
                            console.log(`Adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} (${adapter.info.description})`);
                        }
                    });
                } else {
                    console.log("WebGPU is not supported");
                }
            "#;
            
            if let Some(window) = web_sys::window() {
                let eval_fn = js_sys::Function::new_with_args("code", "return eval(code)");
                let _ = eval_fn.call1(&JsValue::NULL, &JsValue::from_str(js_code));
            }
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.add_srgb_suffix()],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.config.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.8,
                            b: 0.2,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        output.present();

        Ok(())
    }
}

// A simple wrapper to handle state initialization in WebAssembly
#[cfg(target_arch = "wasm32")]
struct StateInitializer {
    window: Arc<Window>,
    app_ptr: *mut App,
}

// Our Apps Initializer that asynchronously starts up, using wasm_bindgen_futures.
#[cfg(target_arch = "wasm32")]
impl StateInitializer {
    fn new(window: Arc<Window>, app: &mut App) -> Self {
        StateInitializer {
            window,
            app_ptr: app as *mut App,
        }
    }

    async fn initialize(self) {
        web_sys::console::log_1(&"Starting state initialization...".into());
        
        // Create the state
        let state = State::new(self.window.clone()).await;
        
        web_sys::console::log_1(&"State initialized, updating App...".into());
        
        // Safety: We know this pointer is valid because the App lives longer than this async task
        unsafe {
            let app = &mut *self.app_ptr;
            app.state = Some(state);
            app.state_initializing = false;
            
            web_sys::console::log_1(&"App state updated!".into());
        }
    }
}

// Our App struct. Defines App state.
#[derive(Default)]
struct App {
    state: Option<State>,
    window: Option<Arc<Window>>,
    #[cfg(target_arch = "wasm32")]
    state_initializing: bool,
}

// Application Handler. This is new required for winit + wgpu newest versions.
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window based on platform
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Multiplatform Window")
                        .with_inner_size(winit::dpi::PhysicalSize::new(DIMX, DIMY))
                )
                .unwrap(),
        );

        window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(DIMX, DIMY)));
        window.set_max_inner_size(Some(winit::dpi::PhysicalSize::new(DIMX, DIMY)));
        window.set_resizable(false);
        
        // Set up canvas for web target - Browser Only
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::console::log_1(&"Setting up web canvas".into());

            let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(DIMX, DIMY));
            
            if let Some(canvas) = window.canvas() {
                let web_window = web_sys::window().unwrap();
                let document = web_window.document().unwrap();
                
                // Try to get element by id "app" first, then fall back to body
                let container = document.get_element_by_id("app")
                    .unwrap_or_else(|| document.body().unwrap().into());
                
                // Explicitly set the canvas dimensions
                canvas.set_width(DIMX.into());
                canvas.set_height(DIMY.into());
                
                // Also set the style to enforce the dimensions in CSS
                let style = canvas.style();
                style.set_property("width", &format!("{}px", DIMX)).unwrap();
                style.set_property("height", &format!("{}px", DIMY)).unwrap();
                style.set_property("max-width", &format!("{}px", DIMX)).unwrap();
                style.set_property("max-height", &format!("{}px", DIMY)).unwrap();
                
                container.append_child(&web_sys::Element::from(canvas))
                    .expect("Couldn't append canvas to document");
                
                web_sys::console::log_1(&"Canvas attached to document".into());
            }
            
            // Store window reference
            self.window = Some(window.clone());
            self.state_initializing = true;
            
            // Begin async initialization
            let initializer = StateInitializer::new(window.clone(), self);
            wasm_bindgen_futures::spawn_local(initializer.initialize());
            
            window.request_redraw();
            return;
        }
        
        // Native platform initialization
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(window.clone()));
            self.state = Some(state);
            self.window = Some(window.clone());
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        {
            // Check if we have a window reference to handle events with
            let window = match &self.window {
                Some(window) => window,
                None => return,
            };
            
            if window.id() != id {
                return;
            }
            
            match event {
                WindowEvent::CloseRequested => {
                    web_sys::console::log_1(&"Close requested".into());
                    event_loop.exit();
                },
                WindowEvent::RedrawRequested => {
                    // If state is initialized, render
                    if let Some(state) = &mut self.state {
                        match state.render() {
                            Ok(_) => {
                                // do nothing for now
                            },
                            Err(wgpu::SurfaceError::Lost) => {
                                web_sys::console::warn_1(&"Surface lost, reconfiguring...".into());
                            },
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                web_sys::console::error_1(&"Out of memory, exiting".into());
                                event_loop.exit();
                            },
                            Err(e) => {
                                web_sys::console::error_1(&format!("Render error: {:?}", e).into());
                            },
                        }
                    } else if self.state_initializing {
                        // If state is still initializing, just log and keep going. Dont hang here
                        web_sys::console::log_1(&"State still initializing, skipping render".into());
                    } else {
                        web_sys::console::log_1(&"No state available for rendering".into());
                    }
                    
                    // Request another frame
                    window.request_redraw();
                },
                WindowEvent::KeyboardInput { event, .. } => {
                    web_sys::console::log_1(&format!("Keyboard Event: {:?}", event).into());
                },
                _ => {
                    web_sys::console::log_1(&format!("Unrecognized Event: {:?}", event).into());
                }
            }
            return;
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            // For native, handle events normally
            let state = match &mut self.state {
                Some(state) => state,
                None => return,
            };
            
            if id != state.window().id() {
                return;
            }
            
            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                },
                WindowEvent::RedrawRequested => {
                    match state.render() {
                        Ok(_) => {},
                        Err(wgpu::SurfaceError::Lost) => println!("Surface lost..."),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => log::error!("render error: {e:?}"),
                    }

                    state.window().request_redraw();
                },
                WindowEvent::KeyboardInput { event, .. } => {
                    println!("Keyboard Event: {:?}", event);
                },
                _ => {
                    println!("Unrecognized Event: {:?}", event);
                }
            }
        }
    }
}

// Multi-platform run function
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
            web_sys::console::log_1(&"Starting web application".into());
        } else {
            env_logger::init();
            log::info!("Starting native application");
        }
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}