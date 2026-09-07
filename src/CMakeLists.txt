add_library(ernie-runtime STATIC
    ernie_gelu.cpp ernie_rmsnorm.cpp ernie_residual.cpp ernie_groupnorm.cpp ernie_attention.cpp
    latent_ops.cpp img2img.cpp image_encoder.cpp component_files.cpp weight_placement.cpp host_memory.cpp weight_session.cpp block_sequence.cpp dit.cpp denoiser.cpp text_encoder.cpp ernie_text_down.cpp conditioning.cpp pe_session.cpp pe_graph.cpp)
target_link_libraries(ernie-runtime PUBLIC ncnn ernie-shape-plan)
target_include_directories(ernie-runtime PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
if(ERNIE_ENABLE_VULKAN)
    include("${PROJECT_SOURCE_DIR}/cmake/ErnieAttentionShader.cmake")
    target_include_directories(ernie-runtime PRIVATE "${ERNIE_NCNN_SOURCE_DIR}/src/layer" "${CMAKE_CURRENT_BINARY_DIR}/generated")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(latent_ops.cpp img2img.cpp denoiser.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

if(ERNIE_BUILD_GENERATOR)
    if(NOT ERNIE_BUILD_TOKENIZER)
        message(FATAL_ERROR "The generator requires ERNIE_BUILD_TOKENIZER=ON")
    endif()
    add_library(ernie-pe STATIC prompt_enhancer.cpp)
    target_link_libraries(ernie-pe PUBLIC ernie-runtime ernie-tokenizer)
    target_include_directories(ernie-pe PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
    add_library(ernie-pipeline STATIC pipeline.cpp model_config.cpp runtime_info.cpp gpu_context.cpp tensor_io.cpp vae.cpp)
    add_library(ernie::pipeline ALIAS ernie-pipeline)
    target_link_libraries(ernie-pipeline PRIVATE ernie-runtime ernie-tokenizer ernie-pe ernie-model-package ernie-execution-metrics)
    target_include_directories(ernie-pipeline PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
    target_compile_features(ernie-pipeline PUBLIC cxx_std_17)
    if(ERNIE_EXPERIMENT_MAPPED_MODEL_LOADING)
        target_compile_definitions(ernie-pipeline PRIVATE ERNIE_EXPERIMENT_MAPPED_MODEL_LOADING=1)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(ernie-pipeline PRIVATE -ffp-contract=off)
    endif()
endif()

# Shared checked geometry and text-padding plan; independent of ncnn.
add_library(ernie-shape-plan STATIC shape_plan.cpp)
target_include_directories(ernie-shape-plan PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

if(ERNIE_BUILD_TOKENIZER)
    # Full graph allowlist used by offline checks and shared-package generation.
    add_library(ernie-shape-graph STATIC shape_graph.cpp model_config.cpp)
    target_include_directories(ernie-shape-graph PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
    target_link_libraries(ernie-shape-graph PUBLIC ernie-tokenizer ernie-shape-plan)
endif()

if(ERNIE_BUILD_TOKENIZER)
    add_library(ernie-model-package STATIC model_package.cpp)
    target_include_directories(ernie-model-package PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
    target_link_libraries(ernie-model-package PUBLIC ernie-shape-graph ernie-tokenizer)
endif()

# Standalone host-side accounting; no ncnn dependency or runtime instrumentation.
add_library(ernie-execution-metrics STATIC execution_metrics.cpp)
target_include_directories(ernie-execution-metrics PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
