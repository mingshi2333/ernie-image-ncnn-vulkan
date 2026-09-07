find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_package(Threads REQUIRED)
set(ERNIE_RUST_TARGET "" CACHE STRING "Cargo target triple; required for cross-compilation")
if(CMAKE_CROSSCOMPILING AND NOT ERNIE_RUST_TARGET)
    message(FATAL_ERROR "Cross-compiling the tokenizer requires ERNIE_RUST_TARGET for the C++ target")
endif()
set(tokenizer_target_arguments)
set(tokenizer_output_directory "${CMAKE_BINARY_DIR}/cargo")
if(ERNIE_RUST_TARGET)
    list(APPEND tokenizer_target_arguments --target "${ERNIE_RUST_TARGET}")
    string(APPEND tokenizer_output_directory "/${ERNIE_RUST_TARGET}")
endif()
set(tokenizer_library "${tokenizer_output_directory}/release/${CMAKE_STATIC_LIBRARY_PREFIX}ernie_tokenizer_bridge${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_custom_command(OUTPUT "${tokenizer_library}"
    COMMAND "${CMAKE_COMMAND}" -E env CARGO_BUILD_JOBS=2 "${CARGO_EXECUTABLE}" build --release --locked
            ${tokenizer_target_arguments}
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
    Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    # System dependencies of the Rust Windows standard library (staticlib).
    target_link_libraries(ernie-tokenizer PRIVATE kernel32 ntdll userenv ws2_32 dbghelp)
else()
    target_link_libraries(ernie-tokenizer PRIVATE m)
endif()
