# WINPUT: A WebAssembly friendly input handler using Rust

To address different input models between web and native:

- Unified Native Input Handling for using SDL2.
- Uses our very own WEVENT Submodule for input event management for consistent ticking and input event translation.
- Is an example of mixing in a previously defined engine submodule.
