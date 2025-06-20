//! LAYER-W Sample Application
//! 
//! This sample application demonstrates the cross-platform capabilities
//! of the LAYER-W framework, including gamepad input handling.

use winput::winput::{WInput};
use std::time::{Duration, Instant};
use std::thread;

// Constants
const TARGET_FPS: u32 = 60;
const MS_PER_FRAME: u64 = 1000 / TARGET_FPS as u64;
const DEBUG_MODE: bool = true;
const PRINT_ALL_EVENTS: bool = true;

fn main() -> Result<(), String> {
    println!("Starting LAYER-W Input Test Application");
    println!("======================================");
    println!("This application demonstrates gamepad input handling.");
    println!("Press Ctrl+C to exit.");
    println!("Target FPS: {}", TARGET_FPS);
    println!();
    
    // Create and initialize the input system with debug mode
    let mut input_system = WInput::new().with_debug(DEBUG_MODE);
    input_system.init()?;
    
    println!("Input system initialized successfully");
    println!("Running main loop...");
    
    // Create a flag to control the loop
    let mut running = true;
    
    // Performance tracking
    let mut frame_count = 0;
    let program_start_time = Instant::now();
    let mut last_fps_report_time = program_start_time;
    let mut last_frame_time = program_start_time;
    
    while running {
        let frame_start = Instant::now();
        
        // Update the input system
        match input_system.update() {
            Ok(_) => {},
            Err(e) => println!("Error updating input system: {}", e),
        }
        
        // Process events
        let mut events_processed = 0;
        while input_system.event_system().has_events() {
            if let Some(event) = input_system.event_system_mut().poll_event() {
                events_processed += 1;
                // Process event...
            }
        }
        
        // FPS calculation - reset every second
        frame_count += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_report_time).as_secs() >= 1 {
            let seconds = now.duration_since(last_fps_report_time).as_secs_f64();
            let fps = frame_count as f64 / seconds;
            
            println!("FPS: {:.2} - Frame time: {:.2}ms - Events processed: {}", 
                fps, 
                frame_start.elapsed().as_millis(), 
                events_processed);
            
            // Important: Reset the counter each second
            frame_count = 0;
            last_fps_report_time = now;
        }
        
        // Sleep to maintain target frame rate
        let frame_duration = frame_start.elapsed();
        if frame_duration < Duration::from_millis(MS_PER_FRAME) {
            thread::sleep(Duration::from_millis(MS_PER_FRAME) - frame_duration);
        }
        
        last_frame_time = now;
    }
    // Cleanup before exiting
    println!("\nShutting down application...");
    input_system.shutdown();
    
    println!("Application terminated successfully");
    Ok(())
}


// For browser WASM
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
#[wasm_bindgen(start)]
pub fn wasm_main() -> Result<(), JsValue> {
    // Set up panic hook for better error messages in WASM
    console_error_panic_hook::set_once();
    
    // Initialize logging for WASM
    console_log::init_with_level(log::Level::Info)
        .expect("Failed to initialize logging in WASM environment");
    
    // Log a message to the browser console
    web_sys::console::log_1(&"LAYER-W initialized in browser".into());
    
    Ok(())
}

// For WASI target
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
fn main() -> Result<(), String> {
    // WASI entry point code
    println!("LAYER-W initialized in WASI environment");
    // Initialize your engine for WASI
    
    Ok(())
}