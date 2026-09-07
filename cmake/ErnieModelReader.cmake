# Keep the pinned checkout intact; compile one reviewed source copy on opt-in.
option(ERNIE_EXPERIMENT_COMPACT_MODEL_READER
    "Correct the pinned ncnn FP16/BF16 temporary vector element count" OFF)

function(ernie_prepare_model_reader source)
    if(NOT ERNIE_EXPERIMENT_COMPACT_MODEL_READER)
        return()
    endif()
    set(original "${source}/src/modelbin.cpp")
    file(READ "${original}" contents)
    # Match Git's canonical source independently of checkout line endings.
    string(REPLACE "\r\n" "\n" contents "${contents}")
    string(SHA256 source_sha "${contents}")
    if(NOT source_sha STREQUAL "eefc538407f88ca9cb3c1c3bde30eb4aa764b1d1c5f4bba9ab9cdd9119c3cb78")
        message(FATAL_ERROR "Compact model reader requires the reviewed pinned modelbin.cpp")
    endif()
    foreach(storage float16 bfloat16)
        # align_data_size is bytes; vector<unsigned short>::resize takes elements.
        string(REPLACE "                ${storage}_weights.resize(align_data_size);"
            "                ${storage}_weights.resize(align_data_size / sizeof(unsigned short));"
            contents "${contents}")
    endforeach()
    set(directory "${CMAKE_BINARY_DIR}/compact-model-reader")
    file(MAKE_DIRECTORY "${directory}")
    file(WRITE "${directory}/modelbin.cpp.in" "${contents}")
    configure_file("${directory}/modelbin.cpp.in" "${directory}/modelbin.cpp" COPYONLY)
    file(WRITE "${directory}/source.sha256" "${source_sha}\n")

    get_target_property(sources ncnn SOURCES)
    set(matches 0)
    set(updated)
    foreach(entry IN LISTS sources)
        if(entry STREQUAL "modelbin.cpp" OR entry STREQUAL "${original}")
            math(EXPR matches "${matches} + 1")
            list(APPEND updated "${directory}/modelbin.cpp")
        else()
            list(APPEND updated "${entry}")
        endif()
    endforeach()
    if(NOT matches EQUAL 1)
        message(FATAL_ERROR "Expected exactly one ncnn ModelBin compilation unit")
    endif()
    set_property(TARGET ncnn PROPERTY SOURCES "${updated}")
endfunction()
