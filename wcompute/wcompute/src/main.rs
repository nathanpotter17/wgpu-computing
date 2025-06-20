#[cfg(not(target_arch = "wasm32"))]
use wcompute::render_demo_cube;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(async {
            if let Err(e) = render_demo_cube().await {
                eprintln!("Demo failed: {}", e);
            }
        });
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        println!("WASM target - entry point is in lib.rs");
    }
}