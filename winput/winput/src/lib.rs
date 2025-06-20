pub mod wevent;
pub mod winput;

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum GenericInputError {
    #[cfg(not(target_arch = "wasm32"))]
    SdlError(String),
    GenericError(String),
}

impl fmt::Display for GenericInputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            GenericInputError::SdlError(msg) => write!(f, "SDL error: {}", msg),
            GenericInputError::GenericError(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for GenericInputError {}

#[cfg(not(target_arch = "wasm32"))]
impl From<String> for GenericInputError {
    fn from(err: String) -> Self {
        GenericInputError::SdlError(err)
    }
}

pub type Result<T> = std::result::Result<T, GenericInputError>;

pub struct Engine {
    /// Event system for handling game events
    event_system: wevent::WEvent,
    /// Input system for handling gamepad input
    input_system: winput::WInput,
    /// Flag indicating if the engine is running
    running: bool,
    /// Debug mode
    debug_mode: bool,
}

impl Engine {
    /// Create a new engine instance
    pub fn new() -> Result<Self> {
        // We need to create the input system with debug enabled
        let mut input_system = winput::WInput::new().with_debug(true);
        input_system.init().map_err(|e| GenericInputError::GenericError(e))?;
        
        Ok(Self {
            event_system: wevent::WEvent::new(),
            input_system,
            running: false,
            debug_mode: true,
        })
    }
    
    /// Initialize the engine with custom settings
    pub fn with_settings(tick_rate: u32, debug_mode: bool) -> Result<Self> {
        // Configure input system with debug mode
        let mut input_system = winput::WInput::new().with_debug(debug_mode);
        input_system.init().map_err(|e| GenericInputError::GenericError(e))?;
        
        // Create with the specified tick rate
        Ok(Self {
            event_system: wevent::WEvent::with_tick_rate(tick_rate),
            input_system,
            running: false,
            debug_mode,
        })
    }
    
    /// Start the engine's main loop
    pub fn start(&mut self) {
        self.running = true;
    }
    
    /// Stop the engine's main loop
    pub fn stop(&mut self) {
        self.running = false;
    }
    
    /// Check if the engine is running
    pub fn is_running(&self) -> bool {
        self.running
    }
    
    /// Update all engine systems (called once per frame)
    pub fn update(&mut self) -> Result<()> {
        // Skip update if engine is not running
        if !self.running {
            return Ok(());
        }
        
        // Update the core event system
        self.event_system.update();
        
        // Update the input system
        self.input_system.update().map_err(|e| GenericInputError::GenericError(e))?;
        
        // Process events from the input system
        self.process_input_events();
        
        // Process events from the main event system
        self.process_main_events();
        
        Ok(())
    }
    
    // Process events from the input system, forward them to the main event system
    fn process_input_events(&mut self) {
        while self.input_system.event_system().has_events() {
            if let Some(event) = self.input_system.event_system_mut().poll_event() {
                self.event_system.push_event(event.event_type.clone(), event.data.clone());
            }
        }
    }
    
    // Process events from the main event system
    fn process_main_events(&mut self) {
        // Process events from the main event system
        while self.event_system.has_events() {
            if let Some(event) = self.event_system.poll_event() {
                match &event.event_type {
                    wevent::EventType::Quit => {
                        println!("Quit event received, stopping engine");
                        self.running = false;
                    },
                    wevent::EventType::Custom(name) => {
                        // Log all custom events for debugging
                        if self.debug_mode {
                            // Use statics to prevent duplicate logging
                            static mut LAST_EVENT_TIME: u64 = 0;
                            static mut LAST_EVENT_NAME: String = String::new();
                            
                            // Only log if this is a new event or if sufficient time has passed
                            let should_log = unsafe {
                                let different_event = *name != LAST_EVENT_NAME;
                                let enough_time_passed = event.timestamp - LAST_EVENT_TIME > 500;
                                
                                if different_event || enough_time_passed {
                                    LAST_EVENT_NAME = name.clone();
                                    LAST_EVENT_TIME = event.timestamp;
                                    true
                                } else {
                                    false
                                }
                            };
                            
                            if should_log {
                                if name.starts_with("button_") {
                                    println!("Button event: {}", name);
                                }
                                else if name.starts_with("gamepad_") {
                                    println!("Gamepad event: {}", name);
                                }
                                else if name.starts_with("axis_") {
                                    // Only log axis events in debug mode
                                    println!("Axis event: {}", name);
                                }
                                else if name.starts_with("unknown_") {
                                    println!("Unknown event: {}", name);
                                }
                                else if name.starts_with("browser_") {     
                                    // Additional logging for WASM
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        use wasm_bindgen::JsValue;
                                        web_sys::console::log_1(&JsValue::from_str(&format!("Browser event: {}", name)));
                                    }
                                }
                                else {
                                    println!("Unrecognized event: {}", name);
                                    
                                    // Additional logging for WASM
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        use wasm_bindgen::JsValue;
                                        web_sys::console::log_1(&JsValue::from_str(&format!("Unrecognized event: {}", name)));
                                    }
                                }
                            }
                        }
                    },
                    _ => {
                        
                    }
                }
            }
        }
    }
    
    /// Access the input system
    pub fn input(&self) -> &winput::WInput {
        &self.input_system
    }
    
    /// Access the input system mutably
    pub fn input_mut(&mut self) -> &mut winput::WInput {
        &mut self.input_system
    }
    
    /// Access the event system
    pub fn events(&self) -> &wevent::WEvent {
        &self.event_system
    }
    
    /// Access the event system mutably
    pub fn events_mut(&mut self) -> &mut wevent::WEvent {
        &mut self.event_system
    }
    
    /// Clean up resources when the engine is shut down
    pub fn shutdown(&mut self) {
        self.input_system.shutdown();
    }
}

// WASM exports for the browser target
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    use super::*;
    
    #[wasm_bindgen]
    pub struct JsEngine {
        inner: Engine,
    }
    
    #[wasm_bindgen]
    impl JsEngine {
        #[wasm_bindgen(constructor)]
        pub fn new() -> std::result::Result<JsEngine, JsValue> {
            
            match Engine::new() {
                Ok(engine) => {
                    web_sys::console::log_1(&JsValue::from_str("Engine created successfully"));
                    Ok(JsEngine { inner: engine })
                },
                Err(e) => {
                    let err_msg = e.to_string();
                    web_sys::console::error_1(&JsValue::from_str(&format!("Error creating engine: {}", err_msg)));
                    Err(JsValue::from_str(&err_msg))
                },
            }
        }
        
        #[wasm_bindgen]
        pub fn start(&mut self) {
            web_sys::console::log_1(&JsValue::from_str("Starting engine..."));
            self.inner.start();
        }
        
        #[wasm_bindgen]
        pub fn stop(&mut self) {
            web_sys::console::log_1(&JsValue::from_str("Stopping engine..."));
            self.inner.stop();
        }
        
        #[wasm_bindgen]
        pub fn is_running(&self) -> bool {
            self.inner.is_running()
        }
        
        #[wasm_bindgen]
        pub fn update(&mut self) -> std::result::Result<(), JsValue> {
            self.inner.update()
                .map_err(|e| {
                    let err_msg = e.to_string();
                    web_sys::console::error_1(&JsValue::from_str(&format!("Error updating engine: {}", err_msg)));
                    JsValue::from_str(&err_msg)
                })
        }
        
        #[wasm_bindgen]
        pub fn is_button_pressed(&self, gamepad_id: u32, button_name: &str) -> bool {
            if let Some(button) = winput::Button::from_str(button_name) {
                self.inner.input().is_button_pressed(gamepad_id, button)
            } else {
                false
            }
        }
        
        #[wasm_bindgen]
        pub fn get_elapsed_time(&self) -> f64 {
            self.inner.input().timer().elapsed_ms() as f64
        }

        #[wasm_bindgen]
        pub fn has_events(&self) -> bool {
            self.inner.events().has_events()
        }
        
    }
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub use self::wasm::JsEngine;