#[cfg(not(target_arch = "wasm32"))]
use wgeneric::run_demo;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(async {
            if let Err(e) = run_demo().await {
                eprintln!("Demo failed: {}", e);
            }
        });
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        println!("WASM target - entry point is in lib.rs");
    }
}