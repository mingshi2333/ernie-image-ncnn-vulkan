option(ERNIE_TEST_VULKAN_VALIDATION "Enable Khronos Vulkan validation for CTest" OFF)

# Call after registering the tests in their CMake directory. CTest examines
# both stdout and stderr; validation errors fail even when a test exits zero.
function(ernie_enable_vulkan_validation)
    if(NOT ERNIE_ENABLE_VULKAN)
        if(ERNIE_TEST_VULKAN_VALIDATION)
            message(FATAL_ERROR "Vulkan test validation requires ERNIE_ENABLE_VULKAN=ON")
        endif()
        return()
    endif()
    get_property(tests DIRECTORY PROPERTY TESTS)
    foreach(test IN LISTS tests)
        set_property(TEST "${test}" APPEND PROPERTY FAIL_REGULAR_EXPRESSION
            "[Vv]alidation [Ee]rror" "VUID-")
        if(ERNIE_TEST_VULKAN_VALIDATION)
            set_property(TEST "${test}" APPEND PROPERTY ENVIRONMENT
                "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation")
        endif()
    endforeach()
    # Preserve each test's existing skip policy for unsupported drivers/BF16.
    # CTest gives return-code skips precedence over failure regexes. CI also
    # runs check_vulkan_validation.py over LastTest.log to cover that case.
endfunction()
