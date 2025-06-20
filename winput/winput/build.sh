#!/bin/bash
# SUBMODULE_002: LAYER-W/ENGINE/winput.rs BUILD SCRIPT

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

wipe_build() {
    if [ -d "target" ]; then
        echo -e "Cleaning previous build...\n"
        rm -rf target
    fi

    if [ -d "wasm" ]; then
        echo -e "Cleaning previous Wasm Bindgen build...\n"
        rm -rf wasm
    fi

    if [ -d "pkg" ]; then
        echo -e "Cleaning previous Wasm Pack build...\n"
        rm -rf pkg
    fi
}

# Main menu
echo -e "\n===== BUILD: SUBMODULE_002: LAYER-W/ENGINE/winput.rs =====\n"
echo -e "Choose build target:"
echo -e "  1) Wasm32 Unknown (wasm32-unknown-unknown)"
echo -e "  2) Wasi (wasm32-wasip2)"
echo -e "  3) Native (cargo, rustc, rustup defaults)"
read -p "Selection [1]: " choice
choice=${choice:-1}

build_native() {
    echo -e "Building for native platform...\n"
    cargo build --release
    confirm "running the native binary"
    cargo run
}

build_wasm32() {
    echo -e "Building for WebAssembly...\n"
    cargo build --release --target=wasm32-unknown-unknown

    echo -e "Do you want to use wasm-bindgen as the build tool? Otherwise, wasm-pack will be used. [Y/n]: "
    read -r use_bindgen
    use_bindgen=${use_bindgen:-Y}

    if [[ "$use_bindgen" =~ ^[Yy]$ ]]; then
        echo -e "Using wasm-bindgen for the build...\n"
        wasm-bindgen target/wasm32-unknown-unknown/release/winput.wasm --out-dir ./wbg --target web
        echo -e "Copying the Wasm Bindgen build to test-runner/wasm directory...\n"
        if [ ! -d "../test-runner/wbg" ]; then
            mkdir -p ../test-runner/wbg
        fi
        cp -r wbg/* ../test-runner/wbg
        rm -rf wbg
        echo -e "See test-runner/index.html via server for the Wasm Bindgen build. It uses the lib.rs\n"
    else
        echo -e "Skipping wasm-bindgen. Using wasm-pack instead.\n"
        wasm-pack build --target web --out-dir wasm
        echo -e "Copying the Wasm Pack build to test-runner/wasm directory...\n"
        if [ ! -d "../test-runner/wmpkg" ]; then
            mkdir -p ../test-runner/wmpkg
        fi
        cp -r wasm/* ../test-runner/wmpkg
        rm -rf wasm
        echo -e "See test-runner/index.html via server for the Wasm Pack build. It uses the lib.rs\n"
    fi
}

build_wasi() {
    echo -e "Building for WASI...\n"
    cargo build --release --target=wasm32-wasip2
    confirm "running the WASI binary with wasmtime (winput-bin.wasm)"
    wasmtime target/wasm32-wasip2/release/winput-bin.wasm
}

case $choice in
    1)
        build_wasm32
        ;;
    2)
        build_wasi
        ;;
    3)
        build_native
        ;;
    *)
        echo -e "Invalid choice. Exiting.\n"
        exit 1
        ;;
esac

echo -e "Build process completed!\n"