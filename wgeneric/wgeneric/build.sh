#!/bin/bash
# LAYER-W ENGINE: LAYER-W/ENGINE/layerw-engine/lib.rs BUILD SCRIPT

set -e  # Exit on error

confirm() {
    local message="$1"
    while true; do
        echo -n "Proceed with $message? [Y/n]: "
        read -r confirm
        case "$confirm" in
            [Yy]) return 0 ;;  # Proceed
            [Nn]) echo "Cancelled."; exit 0 ;;  # Exit
            *) echo "Invalid input. Please enter y or n." ;;
        esac
    done
}

cat << "EOF"
                                         _       
                                        | |      
__      _____ ___  _ __ ___  _ __  _   _| |_ ___ 
\ \ /\ / / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
 \ V  V / (_| (_) | | | | | | |_) | |_| | ||  __/
  \_/\_/ \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                            | |                  
                            |_|                  
EOF
echo -e "================================================================="
echo -e "\n=== WCOMPUTE: LAYER-W/engine/submodules/wcompute/lib.rs ===\n"
echo -e "================================================================="
echo -e "Choose build target:"
echo -e "  1) Wasm32 Unknown (wasm32-unknown-unknown)"
echo -e "  2) Native (cargo, rustc, rustup defaults)"
echo -e "  3) Exit"
read -p "Selection [1/2/3]: " choice
choice=${choice:-1}

exit_early() {
    echo -e "Exiting build script.\n"
    exit 0
}

build_wasm32() {
    echo -e "Building for WebAssembly...\n"
    cargo build --release --target=wasm32-unknown-unknown
    echo -e "\nUsing wasm-bindgen for the build...\n"
    
    if [ ! -d "../test-runner/wbg" ]; then
        mkdir -p ../test-runner/wbg
    fi
    
    wasm-bindgen target/wasm32-unknown-unknown/release/wgeneric.wasm --out-dir ../test-runner/wbg --target web
    
    echo -e "Wasm-bindgen output complete in test-runner/wbg directory.\n"
    echo -e "See test-runner/index.html via VS Code Live Server.\n"
}

build_native() {
    echo -e "Building for native platform...\n"
    cargo run --release
}

case $choice in
    1)
        build_wasm32
        ;;
    2)
        build_native
        ;;
    3)
        exit_early
        ;;
    *)
        echo -e "Invalid choice. Exiting.\n"
        exit 1
        ;;
esac