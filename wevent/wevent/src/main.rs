use std::thread::sleep;
use std::time::Duration;
use wevent::{Timer, WEvent, EventType, EventData, Event};

fn main() {
    println!("Cross-platform event system example");
    
    // Create event system with default 60 FPS tick rate
    let mut event_system = WEvent::new();
    
    println!("Starting event processing...");
    
    // Push some custom events
    event_system.push_event(
        EventType::Custom("GameStart".to_string()),
        Some(EventData::Text("Game initialized".to_string()))
    );
    
    // Simulate game loop for 10 frames
    for i in 0..10 {
        // Update the event system (generates tick events)
        event_system.update();
        
        // Add a custom event every 3rd frame
        if i % 3 == 0 {
            event_system.push_event(
                EventType::Custom("FrameMarker".to_string()),
                Some(EventData::Integer(i))
            );
        }
        
        // Process all events
        while let Some(event) = event_system.poll_event() {
            match event.event_type {
                EventType::Tick => {
                    println!("Frame {}: Tick event at timestamp {}ms", i, event.timestamp);
                },
                EventType::Custom(ref name) => {
                    println!("Frame {}: Custom event '{}' at timestamp {}ms", i, name, event.timestamp);
                    
                    // Access event data if available
                    if let Some(data) = &event.data {
                        match data {
                            EventData::Text(text) => println!("  Data: {}", text),
                            EventData::Integer(value) => println!("  Data: {}", value),
                            _ => println!("  Data: Other type"),
                        }
                    }
                },
                _ => {
                    println!("Frame {}: Other event type at timestamp {}ms", i, event.timestamp);
                }
            }
        }
        
        // Simulate frame work
        sleep(Duration::from_millis(16)); // ~60 FPS
    }
    
    // Add a quit event
    event_system.push_event(EventType::Quit, None);
    
    // Process remaining events including the quit event
    while let Some(event) = event_system.poll_event() {
        match event.event_type {
            EventType::Quit => {
                println!("Quit event received at timestamp {}ms", event.timestamp);
                break;
            },
            _ => {
                println!("Other event at timestamp {}ms", event.timestamp);
            }
        }
    }
    
    println!("Event system example completed");
}