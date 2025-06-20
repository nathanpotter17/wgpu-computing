# WEVENT: A WebAssembly friendly event system using Rust

To address different event models between web and native, and to take a 'page' from Quake 3 (Carmack's Event Journal):

- Create a normalized event queue that collects from all sources
- Implement event translation layers for platform-specific inputs
- Provide a consistent timing model for event handling
