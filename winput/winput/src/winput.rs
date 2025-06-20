use std::collections::HashMap;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use wasm_bindgen::prelude::*;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use web_sys::{Gamepad, GamepadButton, Performance};
use wasm_bindgen::JsValue;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
use sdl2::controller::{GameController, Button as SdlButton};
#[cfg(not(target_arch = "wasm32"))]
use sdl2::GameControllerSubsystem;

use crate::wevent::{Event, EventData, EventType, WEvent, Timer};

// Extend EventType enum to include input-specific events
#[derive(Debug, Clone, PartialEq)]
pub enum InputEventType {
    GamepadConnected(u32),    // Gamepad ID
    GamepadDisconnected(u32), // Gamepad ID
    ButtonPressed(u32, Button), // Gamepad ID, Button
    ButtonReleased(u32, Button), // Gamepad ID, Button
    AxisMotion(u32, u8, i32), // Gamepad ID, Axis ID, Value
    UnknownEvent(String, String), // For any unrecognized events
}

// Map InputEventType to EventType::Custom
impl From<InputEventType> for EventType {
    fn from(input_event: InputEventType) -> Self {
        match input_event {
            InputEventType::GamepadConnected(id) => 
                EventType::Custom(format!("gamepad_connected:{}", id)),
            InputEventType::GamepadDisconnected(id) => 
                EventType::Custom(format!("gamepad_disconnected:{}", id)),
            InputEventType::ButtonPressed(id, button) => 
                EventType::Custom(format!("button_pressed:{}:{}", id, button.to_string())),
            InputEventType::ButtonReleased(id, button) => 
                EventType::Custom(format!("button_released:{}:{}", id, button.to_string())),
            InputEventType::AxisMotion(id, axis, value) =>
                EventType::Custom(format!("axis_motion:{}:{}:{}", id, axis, value)),
            InputEventType::UnknownEvent(name, data) =>
                EventType::Custom(format!("unknown_event:{}:{}", name, data)),
        }
    }
}

// Supported gamepad buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Button {
    A,
    B,
    X,
    Y,
    // Could add more buttons here
}

impl Button {
    pub fn to_string(&self) -> String {
        match self {
            Button::A => "a".to_string(),
            Button::B => "b".to_string(),
            Button::X => "x".to_string(),
            Button::Y => "y".to_string(),
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "a" => Some(Button::A),
            "b" => Some(Button::B),
            "x" => Some(Button::X),
            "y" => Some(Button::Y),
            _ => None,
        }
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    fn from_sdl(button: SdlButton) -> Option<Self> {
        match button {
            SdlButton::A => Some(Button::A),
            SdlButton::B => Some(Button::B),
            SdlButton::X => Some(Button::X),
            SdlButton::Y => Some(Button::Y),
            _ => None,
        }
    }
}

// Common gamepad state tracking structure
#[derive(Clone)]
struct GamepadState {
    id: u32,
    button_states: HashMap<Button, bool>, // true if pressed
    last_button_update_time: u64, // Use timing from wevent
}

impl GamepadState {
    fn new(id: u32) -> Self {
        let mut button_states = HashMap::new();
        button_states.insert(Button::A, false);
        button_states.insert(Button::B, false);
        button_states.insert(Button::X, false);
        button_states.insert(Button::Y, false);
        
        Self {
            id,
            button_states,
            last_button_update_time: 0,
        }
    }
}

// Main input system for handling gamepad input - now with platform-specific fields separated
pub struct WInput {
    // Common fields across all platforms
    event_system: WEvent,
    connected_gamepads: HashMap<u32, GamepadState>,
    timer: Timer,
    debug_mode: bool,
    // Track polling rates
    last_poll_time: u64,
    polling_interval_ms: u64,
    
    // SDL2 specific fields for native platforms
    #[cfg(not(target_arch = "wasm32"))]
    sdl_context: Option<sdl2::Sdl>,
    #[cfg(not(target_arch = "wasm32"))]
    controller_subsystem: Option<GameControllerSubsystem>,
    #[cfg(not(target_arch = "wasm32"))]
    controllers: HashMap<u32, GameController>,
    #[cfg(not(target_arch = "wasm32"))]
    event_pump: Option<sdl2::EventPump>,
    
    // Browser specific fields - fixed to store the navigator.getGamepads function
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    js_gamepad_fn: Option<js_sys::Function>,
}

impl WInput {
    pub fn new() -> Self {
        Self {
            event_system: WEvent::new(),
            connected_gamepads: HashMap::new(),
            timer: Timer::new(),
            debug_mode: true, // Enable debug by default for troubleshooting
            last_poll_time: 0,
            polling_interval_ms: 16, // Default to ~60Hz polling rate
            
            #[cfg(not(target_arch = "wasm32"))]
            sdl_context: None,
            #[cfg(not(target_arch = "wasm32"))]
            controller_subsystem: None,
            #[cfg(not(target_arch = "wasm32"))]
            controllers: HashMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            event_pump: None,
            
            #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
            js_gamepad_fn: None,
        }
    }

    pub fn with_debug(mut self, enable: bool) -> Self {
        self.debug_mode = enable;
        self
    }
    
    pub fn with_polling_rate(mut self, hz: u32) -> Self {
        if hz > 0 {
            self.polling_interval_ms = 1000 / hz as u64;
        }
        self
    }
    
    // Common initialization method with platform-specific implementations
    pub fn init(&mut self) -> Result<(), String> {
        println!("Initializing input system...");
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.init_native()?;
        }
        
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            self.init_browser()?;
        }
        
        #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
        {
            self.init_wasi()?;
        }
        
        Ok(())
    }

    // Native platform initialization
    #[cfg(not(target_arch = "wasm32"))]
    fn init_native(&mut self) -> Result<(), String> {
        // NECESSARY HINT FOR WINDOWS
        sdl2::hint::set("SDL_JOYSTICK_THREAD", "1");
        
        // Initialize SDL for native platforms
        let sdl_context = sdl2::init()?;
        let controller_subsystem = sdl_context.game_controller()?;

        println!("SDL initialized with controller support");
        println!("Event state: {}", controller_subsystem.event_state());
        println!("Controller initialized successfully");

        // Create event pump once during initialization
        let event_pump = sdl_context.event_pump()?;
        self.event_pump = Some(event_pump);

        // Enable controller events
        controller_subsystem.set_event_state(true);
        
        // Open any available game controllers
        let available = controller_subsystem.num_joysticks()?;
        println!("There are {} joysticks available", available);
        
        for id in 0..available {
            if controller_subsystem.is_game_controller(id) {
                match controller_subsystem.open(id) {
                    Ok(controller) => {
                        let instance_id = controller.instance_id();
                        println!("Opened gamepad: {} (instance id: {})", controller.name(), instance_id);
                        self.controllers.insert(instance_id, controller);
                        self.on_gamepad_connected(instance_id);
                    }
                    Err(e) => println!("Failed to open gamepad {}: {}", id, e),
                }
            }
        }
        
        self.sdl_context = Some(sdl_context);
        self.controller_subsystem = Some(controller_subsystem);
        
        Ok(())
    }
    
    // Browser initialization - fixed to store the getGamepads function
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    fn init_browser(&mut self) -> Result<(), String> {
        // For browser, we'll use the Gamepad API
        let window = web_sys::window().ok_or("No global window exists")?;
        let navigator = window.navigator();
        
        // Store the getGamepads function for future use
        let get_gamepads_fn = js_sys::Function::from(
            js_sys::Reflect::get(&navigator, &JsValue::from_str("getGamepads"))
                .map_err(|_| "Failed to get getGamepads function")?
        );
        
        self.js_gamepad_fn = Some(get_gamepads_fn);
        
        println!("Browser gamepad API initialized");
        Ok(())
    }
    
    // WASI initialization
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    fn init_wasi(&mut self) -> Result<(), String> {
        // For WASI, we might need to use the WASI preview APIs
        println!("Initializing WASI gamepad input");
        // Placeholder: In reality, this would depend on WASI future proposals
        
        Ok(())
    }
    
    // Common update method that delegates to platform-specific implementations
    pub fn update(&mut self) -> Result<(), String> {
        // First, let the event system handle its ticks
        self.event_system.update();
        
        // Always poll gamepads on every update for responsive input
        self.poll_gamepads()?;
        
        // Process any pending input events
        self.process_pending_events();
        
        // Safety check: if event queue is getting too large, clear it to prevent memory issues
        if self.event_system.event_count() > 1000 {
            println!("WARNING: Event queue overflow detected ({} events). Clearing queue.", 
                self.event_system.event_count());
            self.event_system.clear_events();
        }
        
        Ok(())
    }

    // Common method to process pending events
    fn process_pending_events(&mut self) {
        // Process a limited number of events per frame to prevent freezing
        let max_events_per_frame = 10;
        let mut events_processed = 0;
        
        while self.event_system.has_events() && events_processed < max_events_per_frame {
            if let Some(event) = self.event_system.poll_event() {
                self.process_event(event);
                events_processed += 1;
            }
        }
        
        if self.debug_mode && events_processed == max_events_per_frame && self.event_system.has_events() {
            println!("WARNING: Not all events processed this frame. Remaining: {}", 
                self.event_system.event_count());
        }
    }
    
    // Poll for gamepad state changes - native implementation
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_gamepads(&mut self) -> Result<(), String> {
        use sdl2::event::Event;
        
        // Collect events first to avoid borrowing issues
        let mut events_to_process = Vec::new();
        
        // Use the event pump created during initialization
        if let Some(event_pump) = &mut self.event_pump {
            for event in event_pump.poll_iter() {
                // Store events to process after releasing the borrow on event_pump
                events_to_process.push(event);
            }
        }
        
        // Process SDL events and convert to our input events
        for event in events_to_process {
            match event {
                Event::ControllerDeviceAdded { which, .. } => {
                    if let Some(controller_subsystem) = &self.controller_subsystem {
                        match controller_subsystem.open(which) {
                            Ok(controller) => {
                                let instance_id = controller.instance_id();
                                println!("Controller connected: {} (instance: {})", controller.name(), instance_id);
                                self.controllers.insert(instance_id, controller);
                                self.on_gamepad_connected(instance_id);
                            }
                            Err(e) => println!("Error opening controller: {}", e),
                        }
                    }
                }
                Event::ControllerDeviceRemoved { which, .. } => {
                    println!("Controller disconnected: {}", which);
                    self.controllers.remove(&which);
                    self.on_gamepad_disconnected(which);
                }
                Event::ControllerButtonDown { which, button, .. } => {
                    if let Some(our_button) = Button::from_sdl(button) {
                        if let Some(gamepad_state) = self.connected_gamepads.get_mut(&which) {
                            gamepad_state.button_states.insert(our_button, true);
                            gamepad_state.last_button_update_time = self.timer.elapsed_ms();
                            
                            let input_event = InputEventType::ButtonPressed(which, our_button);
                            self.push_input_event(input_event, Some(EventData::Text(our_button.to_string())));
                            
                            if self.debug_mode {
                                println!("Button press: {:?} on gamepad {}", our_button, which);
                            }
                        }
                    } else {
                        // Log unfamiliar button
                        let btn_detail = format!("sdl_button:{:?}", button);
                        self.log_unknown_event("unfamiliar_button", &btn_detail);
                    }
                },
                Event::ControllerButtonUp { which, button, .. } => {
                    if let Some(our_button) = Button::from_sdl(button) {
                        if let Some(gamepad_state) = self.connected_gamepads.get_mut(&which) {
                            gamepad_state.button_states.insert(our_button, false);
                            gamepad_state.last_button_update_time = self.timer.elapsed_ms();
                            
                            let input_event = InputEventType::ButtonReleased(which, our_button);
                            self.push_input_event(input_event, Some(EventData::Text(our_button.to_string())));
                            
                            if self.debug_mode {
                                println!("Button release: {:?} on gamepad {}", our_button, which);
                            }
                        }
                    } else {
                        // Log unfamiliar button
                        let btn_detail = format!("sdl_button:{:?}", button);
                        self.log_unknown_event("unfamiliar_button", &btn_detail);
                    }
                },
                Event::ControllerAxisMotion { which, axis, value, .. } => {
                    // Only report significant axis movement to reduce spam
                    if value > 10_000 || value < -10_000 {
                        let input_event = InputEventType::AxisMotion(which, axis as u8, value.into());
                        self.push_input_event(input_event, Some(EventData::Integer(value as i64)));
                        
                        if self.debug_mode {
                            println!("Axis {:?}: {} on gamepad {}", axis, value, which);
                        }
                    }
                },
                _ => {
                    // Handle unknown/unfamiliar events by logging them
                    let event_str = format!("{:?}", event);
                    self.log_unknown_event("sdl_event", &event_str);
                }
            }
        }
        
        // Validate button states against controller state to detect any missed events
        let button_updates = self.check_controller_button_states();
        for (id, button, pressed) in button_updates {
            if let Some(gamepad_state) = self.connected_gamepads.get_mut(&id) {
                gamepad_state.button_states.insert(button, pressed);
                gamepad_state.last_button_update_time = self.timer.elapsed_ms();
                
                let input_event = if pressed {
                    InputEventType::ButtonPressed(id, button)
                } else {
                    InputEventType::ButtonReleased(id, button)
                };
                
                self.push_input_event(input_event, Some(EventData::Text(button.to_string())));
                
                if self.debug_mode {
                    println!("State change detected: {:?} is now {} on gamepad {}", 
                        button, if pressed { "pressed" } else { "released" }, id);
                }
            }
        }
        
        Ok(())
    }
    
    // Helper method to check button states - native implementation
    #[cfg(not(target_arch = "wasm32"))]
    fn check_controller_button_states(&self) -> Vec<(u32, Button, bool)> {
        let mut updates = Vec::new();
        
        for (id, controller) in &self.controllers {
            for button in &[SdlButton::A, SdlButton::B, SdlButton::X, SdlButton::Y] {
                if let Some(our_button) = Button::from_sdl(*button) {
                    let pressed = controller.button(*button);
                    
                    if let Some(gamepad_state) = self.connected_gamepads.get(id) {
                        let prev_pressed = *gamepad_state.button_states.get(&our_button).unwrap_or(&false);
                        
                        // Only generate events if state changed
                        if pressed != prev_pressed {
                            updates.push((*id, our_button, pressed));
                        }
                    }
                }
            }
        }
        
        updates
    }
    
    // Poll for gamepad state changes - browser implementation
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    fn poll_gamepads(&mut self) -> Result<(), String> {
        let window = web_sys::window().ok_or("No global window exists")?;
        let navigator = window.navigator();
        
        // Debug: Check if we can access getGamepads
        if self.js_gamepad_fn.is_none() {
            println!("WARNING: js_gamepad_fn is None");
            return Ok(());
        }
        
        // Call the getGamepads function to get the latest state
        if let Some(get_gamepads_fn) = &self.js_gamepad_fn {
            let gamepads_array = get_gamepads_fn
                .call0(&navigator)
                .map_err(|e| format!("Failed to call getGamepads function: {:?}", e))?;
            
            let gamepads_array = js_sys::Array::from(&gamepads_array);
            
            // Check for connections/disconnections and update button states
            for i in 0..gamepads_array.length() {
                let gamepad_val = gamepads_array.get(i);
                
                // Check if this is null (no gamepad) or undefined
                if gamepad_val.is_null() || gamepad_val.is_undefined() {
                    // This slot has no gamepad
                    let id = i as u32;
                    if self.connected_gamepads.contains_key(&id) {
                        self.on_gamepad_disconnected(id);
                    }
                    continue;
                }
                
                // Use JsCast to convert to Gamepad safely
                if gamepad_val.is_object() && web_sys::Gamepad::instanceof(&gamepad_val) {
                    // Safe to cast since we've checked it's an instance of Gamepad
                    let gamepad = gamepad_val.dyn_into::<web_sys::Gamepad>()
                        .expect("Failed to cast to Gamepad");
                    
                    let id = gamepad.index() as u32;
                    
                    // Check if this is a new connection
                    if !self.connected_gamepads.contains_key(&id) {
                        self.on_gamepad_connected(id);
                    }
                    
                    // Get the gamepad state
                    if let Some(gamepad_state) = self.connected_gamepads.get_mut(&id) {
                        // Map A, B, X, Y to standard gamepad button indices
                        let button_mapping = [
                            (0, Button::A),
                            (1, Button::B),
                            (2, Button::X),
                            (3, Button::Y),
                        ];
                        
                        let buttons = gamepad.buttons();
                        
                        // Collect events to push later
                        let mut events_to_push = Vec::new();

                        // Process familiar buttons
                        for (index, button) in button_mapping.iter() {
                            if (*index as u32) < buttons.length() {
                                // Safely get the button
                                if let Ok(button_val) = js_sys::Reflect::get_u32(&buttons, *index as u32) {
                                    // Only try to process if it's a proper object
                                    if !button_val.is_null() && !button_val.is_undefined() && button_val.is_object() {
                                        // Try to extract pressed state directly
                                        if let Some(pressed) = js_sys::Reflect::get(&button_val, &JsValue::from_str("pressed"))
                                            .ok()
                                            .and_then(|val| val.as_bool()) {
                                            
                                            let prev_pressed = *gamepad_state.button_states.get(button).unwrap_or(&false);
                                            
                                            // Only generate events if state changed
                                            if pressed != prev_pressed {
                                                gamepad_state.button_states.insert(*button, pressed);
                                                gamepad_state.last_button_update_time = self.timer.elapsed_ms();
                                                
                                                let input_event = if pressed {
                                                    InputEventType::ButtonPressed(id, *button)
                                                } else {
                                                    InputEventType::ButtonReleased(id, *button)
                                                };
                                                
                                                events_to_push.push((input_event, EventData::Text(button.to_string())));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Push events after the loop
                        for (input_event, data) in events_to_push {
                            self.push_input_event(input_event, Some(data));
                        }
                        
                        // Process unknown buttons (beyond the first 4)
                        for index in 4..buttons.length() {
                            if let Ok(button_val) = js_sys::Reflect::get_u32(&buttons, index) {
                                if !button_val.is_null() && !button_val.is_undefined() && button_val.is_object() {
                                    if let Some(pressed) = js_sys::Reflect::get(&button_val, &JsValue::from_str("pressed"))
                                        .ok()
                                        .and_then(|val| val.as_bool()) {
                                        
                                        if pressed {
                                            let detail = format!("gamepad_{}_button_{}", id, index);
                                            self.log_unknown_event("browser_gamepad_button", &detail);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Improved axes handling with better type conversion
                        let axes = gamepad.axes();
                        for axis_id in 0..axes.length() {
                            // Get axis value directly as f64 instead of JsValue reflection
                            if let Some(value) = js_sys::Reflect::get_u32(&axes, axis_id).ok()
                                .and_then(|val| val.as_f64()) {
                                
                                // Only report significant axis movement (value is between -1.0 and 1.0 in browser)
                                if value.abs() > 0.1 {
                                    // Scale to match SDL2 range (-32768 to 32767)
                                    let scaled_value = (value * 32767.0) as i32;
                                    
                                    // Log unknown axes beyond standard ones (usually 4-6 axes for modern gamepads)
                                    if axis_id >= 6 {
                                        let detail = format!("axis_{}:{:.3}", axis_id, value);
                                        self.log_unknown_event("browser_gamepad_axis", &detail);
                                    }
                                    
                                    let input_event = InputEventType::AxisMotion(id, axis_id as u8, scaled_value);
                                    self.push_input_event(input_event, Some(EventData::Float(value)));
                                    
                                    if self.debug_mode {
                                        println!("Axis {}: {} on gamepad {}", axis_id, value, id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Poll for gamepad state changes - WASI implementation
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    fn poll_gamepads(&mut self) -> Result<(), String> {
        // Placeholder for WASI implementation
        // For testing purposes, connect a virtual gamepad if none are connected
        if self.connected_gamepads.is_empty() {
            self.on_gamepad_connected(0);
        }
        
        Ok(())
    }

    fn push_input_event(&mut self, input_event: InputEventType, data: Option<EventData>) {
        let timestamp = self.timer.elapsed_ms();
        let custom_event: EventType = input_event.into();
        
        let event = Event {
            event_type: custom_event,
            timestamp,
            data,
        };
        
        self.event_system.push_event(event.event_type.clone(), event.data.clone());
    }

    fn log_unknown_event(&mut self, category: &str, detail: &str) {
        let timestamp = self.timer.elapsed_ms();
        let formatted_detail = format!("{}@{}ms", detail, timestamp);
        let input_event = InputEventType::UnknownEvent(category.to_string(), formatted_detail);
        self.push_input_event(input_event, None);
        
        // Always log unfamiliar events to console, even if debug mode is off
        println!("UNKNOWN EVENT: {} - {}", category, detail);
    }
    
    // Helper method for warnings
    fn log_warning(&self, message: &str) {
        println!("WARNING: {}", message);
        
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsValue;
            web_sys::console::log_1(&JsValue::from_str(&format!("WARNING: {}", message)));
        }
    }
    
    // Common method for handling gamepad connection
    fn on_gamepad_connected(&mut self, id: u32) {
        // Create a new gamepad state
        let gamepad_state = GamepadState::new(id);
        self.connected_gamepads.insert(id, gamepad_state);
        
        // Push connection event
        let input_event = InputEventType::GamepadConnected(id);
        self.event_system.push_event(
            input_event.into(),
            Some(EventData::Integer(id as i64)),
        );
        
        println!("Gamepad {} connected", id);
    }
    
    // Common method for handling gamepad disconnection
    fn on_gamepad_disconnected(&mut self, id: u32) {
        // Remove gamepad state
        self.connected_gamepads.remove(&id);
        
        // Push disconnection event
        let input_event = InputEventType::GamepadDisconnected(id);
        self.event_system.push_event(
            input_event.into(),
            Some(EventData::Integer(id as i64)),
        );
        
        println!("Gamepad {} disconnected", id);
    }
    
    // Track last event times to prevent duplicates
    fn process_event(&mut self, event: Event) {
        match &event.event_type {
            EventType::Custom(name) => {
                if self.debug_mode {
                    web_sys::console::log_1(&JsValue::from_str(&format!("Event System (RS): {}", name)));
                }
                
                if name.starts_with("button_pressed:") {
                    // Log only in debug mode
                    if self.debug_mode {
                        // Only log once per event, using timestamp to prevent duplication
                        static mut LAST_PRESSED_TIME: u64 = 0;
                        static mut LAST_PRESSED_EVENT: String = String::new();
                        
                        // Safety: we only access these statics in this function
                        unsafe {
                            // Only log if this is a different event or if enough time has passed
                            if *name != LAST_PRESSED_EVENT || event.timestamp - LAST_PRESSED_TIME > 500 {
                                println!("INPUT EVENT: {}", name);
                                LAST_PRESSED_EVENT = name.clone();
                                LAST_PRESSED_TIME = event.timestamp;
                                
                                // Extract button for more visible output
                                if name.contains(":a") {
                                    println!("A BUTTON PRESSED");
                                } else if name.contains(":b") {
                                    println!("B BUTTON PRESSED");
                                } else if name.contains(":x") {
                                    println!("X BUTTON PRESSED");
                                } else if name.contains(":y") {
                                    println!("Y BUTTON PRESSED");
                                }
                            }
                        }
                    }
                } 
                else if name.starts_with("button_released:") {
                    // Log only in debug mode
                    if self.debug_mode {
                        // Only log once per event, using timestamp to prevent duplication
                        static mut LAST_RELEASED_TIME: u64 = 0;
                        static mut LAST_RELEASED_EVENT: String = String::new();
                        
                        // Safety: we only access these statics in this function
                        unsafe {
                            // Only log if this is a different event or if enough time has passed
                            if *name != LAST_RELEASED_EVENT || event.timestamp - LAST_RELEASED_TIME > 500 {
                                println!("INPUT EVENT: {}", name);
                                LAST_RELEASED_EVENT = name.clone();
                                LAST_RELEASED_TIME = event.timestamp;
                                
                                // Extract button for more visible output
                                if name.contains(":a") {
                                    println!("A BUTTON RELEASED");
                                } else if name.contains(":b") {
                                    println!("B BUTTON RELEASED");
                                } else if name.contains(":x") {
                                    println!("X BUTTON RELEASED");
                                } else if name.contains(":y") {
                                    println!("Y BUTTON RELEASED");
                                }
                            }
                        }
                    }
                }
                else if name.starts_with("gamepad_connected:") {
                    println!("GAMEPAD CONNECTED: {}", name);
                }
                else if name.starts_with("gamepad_disconnected:") {
                    println!("GAMEPAD DISCONNECTED: {}", name);
                }
                else if name.starts_with("axis_motion:") {
                    if self.debug_mode {
                        println!("AXIS MOTION EVENT: {}", name);
                    }
                }
            }
            EventType::Tick => {
                // Handle tick events if needed
            }
            EventType::Quit => {
                println!("Quit event received");
            }
        }
    }
    
    // Public API for checking button state
    pub fn is_button_pressed(&self, gamepad_id: u32, button: Button) -> bool {
        if let Some(gamepad) = self.connected_gamepads.get(&gamepad_id) {
            return *gamepad.button_states.get(&button).unwrap_or(&false);
        }
        false
    }
    
    // Get a reference to the underlying event system
    pub fn event_system(&self) -> &WEvent {
        &self.event_system
    }
    
    // Get a mutable reference to the underlying event system
    pub fn event_system_mut(&mut self) -> &mut WEvent {
        &mut self.event_system
    }

    pub fn timer(&self) -> &Timer {
        &self.timer
    }
    
    // Clean up resources
    pub fn shutdown(&mut self) {
        println!("Shutting down input system...");
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            // SDL controllers are automatically closed when dropped
            self.controllers.clear();
        }
        
        // Clear any remaining events
        self.event_system.clear_events();
    }
}

// === WASM Bindings for Browser ===
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
#[wasm_bindgen]
pub struct JsWInput {
    inner: WInput,
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
#[wasm_bindgen]
impl JsWInput {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut input = WInput::new();
        input.init().expect("Failed to initialize input system");
        Self { inner: input }
    }
    
    #[wasm_bindgen]
    pub fn update(&mut self) {
        self.inner.update().expect("Failed to update input system");
    }
    
    #[wasm_bindgen]
    pub fn is_button_pressed(&self, gamepad_id: u32, button_name: &str) -> bool {
        if let Some(button) = Button::from_str(button_name) {
            self.inner.is_button_pressed(gamepad_id, button)
        } else {
            false
        }
    }

    #[wasm_bindgen]
    pub fn get_elapsed_ms(&self) -> f64 {
        self.inner.timer().elapsed_ms() as f64
    }
    
    #[wasm_bindgen]
    pub fn has_events(&self) -> bool {
        self.inner.event_system().has_events()
    }
}