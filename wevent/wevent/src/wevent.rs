use std::time::{Duration, Instant};
use std::collections::VecDeque;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    Tick,
    Quit,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct Event {
    pub event_type: EventType,
    pub timestamp: u64, // milliseconds since epoch or start
    pub data: Option<EventData>,
}

#[derive(Debug, Clone)]
pub enum EventData {
    None,
    Integer(i64),
    Float(f64),
    Text(String),
}

// Cross-platform timer.
pub struct Timer {
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
    #[cfg(target_arch = "wasm32")]
    start_time_ms: f64, // milliseconds, since `Date::now()` returns ms as f64
}

// Cross-platform event system for handling events in a game engine.
pub struct WEvent {
    event_queue: VecDeque<Event>,
    timer: Timer,
    last_tick_time: u64,
    tick_interval_ms: u64,
}

impl Timer {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                start: Instant::now(),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                start_time_ms: Self::now_ms(),
            }
        }
    }

    pub fn elapsed(&self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed()
        }

        #[cfg(target_arch = "wasm32")]
        {
            let now_ms = Self::now_ms();
            let elapsed_ms = now_ms - self.start_time_ms;
            Duration::from_secs_f64(elapsed_ms / 1000.0)
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }

    pub fn reset(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start = Instant::now();
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.start_time_ms = Self::now_ms();
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn now_ms() -> f64 {
        use js_sys::Date;
        Date::now()
    }
}

impl WEvent {
    pub fn new() -> Self {
        Self {
            event_queue: VecDeque::new(),
            timer: Timer::new(),
            last_tick_time: 0,
            tick_interval_ms: 16, // ~60 FPS by default
        }
    }

    pub fn with_tick_rate(fps: u32) -> Self {
        let tick_interval_ms = if fps > 0 { 1000_u64 / fps as u64 } else { 16 };
        Self {
            event_queue: VecDeque::new(),
            timer: Timer::new(),
            last_tick_time: 0,
            tick_interval_ms,
        }
    }

    // Push a new event to the queue
    pub fn push_event(&mut self, event_type: EventType, data: Option<EventData>) {
        let event = Event {
            event_type,
            timestamp: self.timer.elapsed_ms(),
            data,
        };
        self.event_queue.push_back(event);
    }

    // Get the next event from the queue
    pub fn poll_event(&mut self) -> Option<Event> {
        self.event_queue.pop_front()
    }

    // Process tick events based on configured interval
    pub fn update(&mut self) {
        let current_time = self.timer.elapsed_ms();
        
        // Generate tick event if enough time has passed
        if current_time - self.last_tick_time >= self.tick_interval_ms {
            self.last_tick_time = current_time;
            self.push_event(EventType::Tick, Some(EventData::Integer(current_time as i64)));
        }
    }

    // Clear all pending events
    pub fn clear_events(&mut self) {
        self.event_queue.clear();
    }

    // Check if there are any events in the queue
    pub fn has_events(&self) -> bool {
        !self.event_queue.is_empty()
    }

    // Get number of events in the queue
    pub fn event_count(&self) -> usize {
        self.event_queue.len()
    }

    // Set a new tick interval in milliseconds
    pub fn set_tick_interval(&mut self, interval_ms: u64) {
        self.tick_interval_ms = interval_ms;
    }

    // Get the current timer instance
    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    // Get a mutable reference to the timer
    pub fn timer_mut(&mut self) -> &mut Timer {
        &mut self.timer
    }
}

// === WASM Bindings for Browser ===
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct JsTimer {
    inner: Timer,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl JsTimer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsTimer {
        JsTimer {
            inner: Timer::new(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.inner.elapsed().as_secs_f64() * 1000.0
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct JsWEvent {
    inner: WEvent,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl JsWEvent {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WEvent::new(),
        }
    }

    #[wasm_bindgen]
    pub fn update(&mut self) {
        self.inner.update();
    }

    #[wasm_bindgen]
    pub fn has_events(&self) -> bool {
        self.inner.has_events()
    }

    #[wasm_bindgen]
    pub fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    #[wasm_bindgen]
    pub fn push_custom_event(&mut self, name: &str) {
        self.inner.push_event(
            EventType::Custom(name.to_string()),
            Some(EventData::Text(name.to_string())),
        );
    }

    #[wasm_bindgen]
    pub fn clear_events(&mut self) {
        self.inner.clear_events();
    }
}