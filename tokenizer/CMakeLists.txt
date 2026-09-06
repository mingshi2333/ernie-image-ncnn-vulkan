find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_package(Threads REQUIRED)
set(tokenizer_library "${CMAKE_BINARY_DIR}/cargo/release/libernie_tokenizer_bridge.a")
add_custom_command(OUTPUT "${tokenizer_library}"
    COMMAND "${CMAKE_COMMAND}" -E env CARGO_BUILD_JOBS=2 "${CARGO_EXECUTABLE}" build --release --locked
            --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" --target-dir "${CMAKE_BINARY_DIR}/cargo"
    DEPENDS Cargo.toml Cargo.lock src/lib.rs src/package.rs src/shared_package.rs schema3_contract.json "${PROJECT_SOURCE_DIR}/sources.lock.json"
    COMMENT "Building pinned native Tokenizers bridge")
add_custom_target(ernie-tokenizer-rust DEPENDS "${tokenizer_library}")
add_library(ernie-tokenizer-native STATIC IMPORTED GLOBAL)
set_target_properties(ernie-tokenizer-native PROPERTIES IMPORTED_LOCATION "${tokenizer_library}")
add_dependencies(ernie-tokenizer-native ernie-tokenizer-rust)
add_library(ernie-tokenizer STATIC "${PROJECT_SOURCE_DIR}/src/tokenizer.cpp")
target_include_directories(ernie-tokenizer PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")
target_link_libraries(ernie-tokenizer PRIVATE
    "$<BUILD_INTERFACE:ernie-tokenizer-native>"
    "$<INSTALL_INTERFACE:ernie::tokenizer-native>"
    Threads::Threads ${CMAKE_DL_LIBS} m)
