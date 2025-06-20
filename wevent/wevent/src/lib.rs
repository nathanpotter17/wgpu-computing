mod wevent;

pub use wevent::Timer;
pub use wevent::WEvent;
pub use wevent::Event;
pub use wevent::EventType;
pub use wevent::EventData;

#[cfg(target_arch = "wasm32")]
pub use wevent::JsTimer;
#[cfg(target_arch = "wasm32")]
pub use wevent::JsWEvent;

#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    use super::{JsTimer, JsWEvent};
    
    #[wasm_bindgen]
    pub fn create_timer() -> JsTimer {
        JsTimer::new()
    }
    
    #[wasm_bindgen]
    pub fn create_event_system() -> JsWEvent {
        JsWEvent::new()
    }
    
    #[wasm_bindgen]
    pub fn run_demo() -> String {
        let timer = JsTimer::new();
        
        // Simulate some work
        let mut result: u64 = 0;
        for i in 0..1_000_000 {
            result = i + 1;
        }
        
        let elapsed = timer.elapsed_ms();
        format!("Calculation result: {}, took: {}ms", result, elapsed)
    }
    
    #[wasm_bindgen(start)]
    pub fn wasm_start() {
        #[cfg(debug_assertions)]
        console_error_panic_hook::set_once();
    }
}