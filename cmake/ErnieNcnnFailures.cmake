function(ernie_prepare_ncnn_failures source)
    if(NOT ERNIE_ENABLE_VULKAN)
        return()
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(directory "${CMAKE_BINARY_DIR}/ncnn-vulkan-failures")
    set(allocator "${CMAKE_BINARY_DIR}/ncnn-weight-buffer/allocator.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${source}/src/command.cpp" "${source}/src/net.cpp"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/derive_ncnn_failures.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ncnn_vulkan_failure.h")
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/derive_ncnn_failures.py"
        "${source}" "${allocator}" "${directory}"
        OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY)
    get_target_property(sources ncnn SOURCES)
    foreach(unit allocator command net)
        set(updated)
        set(matches 0)
        foreach(entry IN LISTS sources)
            if(entry STREQUAL "${unit}.cpp" OR entry STREQUAL "${source}/src/${unit}.cpp"
               OR (unit STREQUAL "allocator" AND entry STREQUAL "${allocator}"))
                math(EXPR matches "${matches} + 1")
                list(APPEND updated "${directory}/${unit}.cpp")
            else()
                list(APPEND updated "${entry}")
            endif()
        endforeach()
        if(NOT matches EQUAL 1)
            message(FATAL_ERROR "Expected one ncnn ${unit} compilation unit for failure propagation")
        endif()
        set(sources "${updated}")
    endforeach()
    set_property(TARGET ncnn PROPERTY SOURCES "${sources}")
endfunction()
