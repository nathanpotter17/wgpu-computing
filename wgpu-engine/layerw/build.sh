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
      ...                                                            ...    .     ...
  .zf"` `"tu                 ..                                   .~`"888x.!**h.-``888h.
 x88      '8N.              @L                        .u    .    dX   `8888   :X   48888>
 888k     d88&       u     9888i   .dL       .u     .d88B :@8c  '888x  8888  X88.  '8888>
 8888N.  @888F    us888u.  `Y888k:*888.   ud8888.  ="8888f8888r '88888 8888X:8888:   )?""`
 `88888 9888%  .@88 "8888"   888E  888I :888'8888.   4888>'88"   `8888>8888 '88888>.88h.
   %888 "88F   9888  9888    888E  888I d888 '88%"   4888> '       `8" 888f  `8888>X88888.
    8"   "*h=~ 9888  9888    888E  888I 8888.+"      4888>        -~` '8%"     88" `88888X
  z8Weu        9888  9888    888E  888I 8888L       .d888L .+     .H888n.      XHn.  `*88!
 ""88888i.   Z 9888  9888   x888N><888' '8888c. .+  ^"8888*"     :88888888x..x88888X.  `!
"   "8888888*  "888*""888"   "88"  888   "88888%       "Y"       f  ^%888888% `*88888nx"
      ^"**""    ^Y"   ^Y'          88F     "YP'                       `"**"`    `"**""
                                  98"
                                ./"
                               ~`
EOF
echo -e "================================================================="
echo -e "\n===== LAYER-W ENGINE: LAYER-W/engine/layerw/lib.rs =======\n"
echo -e "================================================================="
echo -e "Choose build target:"
echo -e "  1) Wasm32 Unknown Release (wasm32-unknown-unknown)"
echo -e "  2) Native Quick Build (cargo, rustc, rustup defaults)"
echo -e "  3) Build Native Release                  " 
echo -e "  4) Watch mode (hot reload for development)"
echo -e "  5) Exit"
read -p "Selection [1/2/3/4/5]: " choice
choice=${choice:-1}

exit_early() {
    echo -e "Exiting build script.\n"
    exit 0
}

build_wasm32() {
    echo -e "Building for WebAssembly...\n"
    cargo build --release --target=wasm32-unknown-unknown
    echo -e "\nUsing wasm-bindgen for the build...\n"
    wasm-bindgen target/wasm32-unknown-unknown/release/layerw.wasm --out-dir ./wbg --target web
    echo -e "Copying the Wasm Bindgen build to test-runner/wbg directory...\n"
    if [ ! -d "test-runner/wbg" ]; then
        mkdir -p test-runner/wbg
    fi
    cp -r wbg/* test-runner/wbg
    rm -rf wbg
    echo -e "See test-runner/index.html via VS Code Live Server.\n"
}

build_native() {
    echo -e "Building for native platform...\n"
    cargo run
}

build_native_release() {
    echo -e "Building for native platform...\n"
    cargo run --release
}

native_hot_reload() {
    echo -e "Starting native watch mode...\n"
    cargo watch -w src -w Cargo.toml -x "run --release"
}

wasm_hot_reload() {
    echo -e "Starting Wasm watch mode...\n"
    cargo watch -w src -w Cargo.toml -x "build --release --target=wasm32-unknown-unknown" \
        -x "wasm-bindgen target/wasm32-unknown-unknown/release/layerw.wasm --out-dir ./wbg --target web" \
        -x "cp -r wbg/* test-runner/wbg" \
        -x "rm -rf wbg"
    echo -e "See test-runner/index.html via VS Code Live Server.\n"
}

watch_mode() {
    confirm "watch mode"
    echo -e "Choose watch mode target:"
    echo -e "  1) Wasm32 Unknown (wasm32-unknown-unknown)"
    echo -e "  2) Native (cargo, rustc, rustup defaults)"
    read -p "Selection [1/2]: " watch_choice
    watch_choice=${watch_choice:-1}
    case $watch_choice in
        1)
            wasm_hot_reload
            ;;
        2)
            native_hot_reload
            ;;
        *)
            echo -e "Invalid choice. Exiting.\n"
            exit 1
            ;;
    esac
    return 0
}

case $choice in
    1)
        build_wasm32
        ;;
    2)
        build_native
        ;;
    3)
        build_native_release
        ;;
    4)  watch_mode
        ;;
    5)
        exit_early
        ;;
    *)
        echo -e "Invalid choice. Exiting.\n"
        exit 1
        ;;
esac
