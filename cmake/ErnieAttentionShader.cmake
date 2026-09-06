# Retain the pinned ncnn SDPA tiling, bindings and layout handling. Change only
# the long softmax and probability/value summations. Verify the complete source
# before deriving a shader, so an upstream update cannot silently misapply it.
set(sdpa_source "${ERNIE_NCNN_SOURCE_DIR}/src/layer/vulkan/shader/sdpa_cross.comp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${sdpa_source}")
file(SHA256 "${sdpa_source}" sdpa_source_hash)
if(NOT sdpa_source_hash STREQUAL "a6e6d5caaa411e4253a01a57b5fef461eeb1d9e634e3043920b9c6ad7dca4b14")
    message(FATAL_ERROR "Review the changed ncnn SDPA shader before deriving the ERNIE accumulation shader")
endif()
file(READ "${sdpa_source}" ERNIE_SDPA_SHADER_SOURCE)
set(compensated_function [=[
// Kahan accumulation: all four independent output rows retain their low bits.
// Explicit precise temporaries forbid reassociation which would erase the
// compensation. Products remain FP32, like the original non-cooperative path.
void ernie_accumulate(afp a, afpvec4 b, inout afpvec4 sum, inout afpvec4 correction)
{
    precise afpvec4 product = a * b;
    precise afpvec4 adjusted = product - correction;
    precise afpvec4 next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}
]=])
string(REPLACE "void main()" "${compensated_function}\nvoid main()" ERNIE_SDPA_SHADER_SOURCE "${ERNIE_SDPA_SHADER_SOURCE}")
set(components r g b a)
foreach(i RANGE 0 3)
    list(GET components ${i} component)
    string(REPLACE "afpvec4 sum${i} = afpvec4(0.f);"
        "precise afpvec4 sum${i} = afpvec4(0.f);\n    afpvec4 correction${i} = afpvec4(0.f);"
        ERNIE_SDPA_SHADER_SOURCE "${ERNIE_SDPA_SHADER_SOURCE}")
    string(REPLACE "sum${i} += a.${component} * b;"
        "ernie_accumulate(a.${component}, b, sum${i}, correction${i});"
        ERNIE_SDPA_SHADER_SOURCE "${ERNIE_SDPA_SHADER_SOURCE}")
endforeach()
set(softmax_source "${ERNIE_NCNN_SOURCE_DIR}/src/layer/vulkan/shader/softmax_reduce_sum.comp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${softmax_source}")
file(SHA256 "${softmax_source}" softmax_source_hash)
if(NOT softmax_source_hash STREQUAL "8be03bc935b53d7f8fb7e4900196bcc7c660eccc64c8f2b3ec78c7986b7873dd")
    message(FATAL_ERROR "Review the changed ncnn softmax shader before deriving the ERNIE accumulation shader")
endif()
file(READ "${softmax_source}" ERNIE_SOFTMAX_SHADER_SOURCE)
string(REPLACE "afp sum_value = afp(0.f);"
    "precise afp sum_value = afp(0.f);\n    afp correction = afp(0.f);"
    ERNIE_SOFTMAX_SHADER_SOURCE "${ERNIE_SOFTMAX_SHADER_SOURCE}")
string(REPLACE "sum_value += v;"
    "precise afp adjusted = v - correction;\n            precise afp next = sum_value + adjusted;\n            correction = (next - sum_value) - adjusted;\n            sum_value = next;"
    ERNIE_SOFTMAX_SHADER_SOURCE "${ERNIE_SOFTMAX_SHADER_SOURCE}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/sdpa_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa_shader.h" @ONLY)
