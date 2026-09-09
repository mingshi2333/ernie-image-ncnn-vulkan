# The pinned Flash shader packs probabilities through a BF16 accumulator type.
# BF16 cooperative A/B support does not imply BF16 accumulator support. Select
# ncnn's ordinary BF16 Flash path until that shader has a reviewed packing fix.
function(ernie_prepare_ncnn_bf16_sdpa source)
    if(NOT ERNIE_ENABLE_VULKAN)
        return()
    endif()
    set(original "${source}/src/layer/vulkan/sdpa_vulkan.cpp")
    set(shader "${source}/src/layer/vulkan/shader/sdpa_fa_cm.comp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${original}" "${shader}")
    file(READ "${original}" contents)
    file(READ "${shader}" shader_contents)
    string(REPLACE "\r\n" "\n" contents "${contents}")
    string(REPLACE "\r\n" "\n" shader_contents "${shader_contents}")
    string(SHA256 source_sha "${contents}")
    string(SHA256 shader_sha "${shader_contents}")
    if(NOT source_sha STREQUAL "933b6bc354d3474b543d3716c224fa7ef544d5fddcf48b5eeebccf3059c00553"
       OR NOT shader_sha STREQUAL "956b83ef564b683f53ecb6f41d51416ef08bc306b7abb86a74a0188b82387743")
        message(FATAL_ERROR "Review BF16 SDPA and its cooperative shader before applying the compatibility fallback")
    endif()
    set(before [=[    if (vkdev->info.support_bf16_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_bf16_storage)
    {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }]=])
    set(after [=[    // The pinned sdpa_fa_cm shader constructs BF16 accumulator matrices
    // while packing probabilities. BF16 A/B with FP32 accumulation does not
    // support those temporary types. Keep native BF16 Flash attention enabled;
    // only its cooperative variant is disabled. FP16/FP32 are unchanged.
    if (opt.use_bf16_storage)
        use_cooperative_matrix = false;]=])
    string(FIND "${contents}" "${before}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "Expected BF16 SDPA cooperative selection is missing")
    endif()
    string(REPLACE "${before}" "${after}" contents "${contents}")
    # The copy is outside layer/vulkan; retain ncnn's existing include roots.
    string(REPLACE "#include \"sdpa_vulkan.h\"" "#include \"layer/vulkan/sdpa_vulkan.h\"" contents "${contents}")
    string(SHA256 derived_sha "${contents}")
    set(directory "${CMAKE_BINARY_DIR}/ncnn-bf16-sdpa")
    file(MAKE_DIRECTORY "${directory}")
    file(WRITE "${directory}/sdpa_vulkan.cpp.in" "${contents}")
    configure_file("${directory}/sdpa_vulkan.cpp.in" "${directory}/sdpa_vulkan.cpp" COPYONLY)
    file(WRITE "${directory}/source.sha256" "${source_sha}\n")
    file(WRITE "${directory}/shader.sha256" "${shader_sha}\n")
    file(WRITE "${directory}/derived.sha256" "${derived_sha}\n")
    get_target_property(sources ncnn SOURCES)
    set(matches 0)
    set(updated)
    foreach(entry IN LISTS sources)
        if(entry STREQUAL "layer/vulkan/sdpa_vulkan.cpp" OR entry STREQUAL "${original}")
            math(EXPR matches "${matches} + 1")
            list(APPEND updated "${directory}/sdpa_vulkan.cpp")
        else()
            list(APPEND updated "${entry}")
        endif()
    endforeach()
    if(NOT matches EQUAL 1)
        message(FATAL_ERROR "Expected exactly one ncnn Vulkan SDPA compilation unit")
    endif()
    set_property(TARGET ncnn PROPERTY SOURCES "${updated}")
endfunction()
