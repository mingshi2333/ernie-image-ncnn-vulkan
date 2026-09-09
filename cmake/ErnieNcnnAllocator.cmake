# Repair the reviewed ncnn weight-buffer contract in a build-directory copy.
# Replacing the ncnn compilation unit also fixes its installed SDK archive.
function(ernie_prepare_ncnn_allocator source)
    if(NOT ERNIE_ENABLE_VULKAN)
        return()
    endif()
    set(original "${source}/src/allocator.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${original}")
    file(READ "${original}" contents)
    # Authenticate the Git source independently of Windows checkout line endings.
    string(REPLACE "\r\n" "\n" contents "${contents}")
    string(SHA256 source_sha "${contents}")
    if(ERNIE_ENABLE_ALLOCATION_METRICS)
        # derive_allocation_ncnn.py first authenticates the complete source tree.
        # This additionally fixes the exact reviewed instrumentation composition.
        set(expected "f3452248872cdebd7535c8d6e752c99c600758beb54a363f03a0a77498cc4bce")
    else()
        set(expected "601d69dab40823366fa6aa2be00c8e37bb0f1e960fe96323c36daed887c4fda2")
    endif()
    if(NOT source_sha STREQUAL expected)
        message(FATAL_ERROR "Review the ncnn allocator before applying the weight-buffer compatibility fix: ${source_sha}")
    endif()

    set(before [=[    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);]=])
    set(after [=[    // Imported host memory requires the same handle type on its buffer.
    // Weight layout conversions also use this buffer as a transfer source.
    VkExternalMemoryBufferCreateInfo externalBufferInfo = {};
    externalBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
#if !defined(_WIN32)
    if (d->prefer_host_memory && vkdev->info.support_VK_EXT_external_memory_host())
        bufferInfo.pNext = &externalBufferInfo;
#endif
    bufferInfo.size = new_block_size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult bufferResult = vkCreateBuffer(vkdev->vkdevice(), &bufferInfo, 0, &block->buffer);
    if (bufferResult != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateBuffer for weights failed %d", bufferResult);
        delete block;
        return 0;
    }]=])
    string(FIND "${contents}" "${before}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "Expected ncnn weight-buffer construction is missing")
    endif()
    string(REPLACE "${before}" "${after}" contents "${contents}")
    string(SHA256 patched_sha "${contents}")
    set(directory "${CMAKE_BINARY_DIR}/ncnn-weight-buffer")
    file(MAKE_DIRECTORY "${directory}")
    file(WRITE "${directory}/allocator.cpp.in" "${contents}")
    configure_file("${directory}/allocator.cpp.in" "${directory}/allocator.cpp" COPYONLY)
    file(WRITE "${directory}/source.sha256" "${source_sha}\n")
    file(WRITE "${directory}/patched.sha256" "${patched_sha}\n")

    get_target_property(sources ncnn SOURCES)
    set(matches 0)
    set(updated)
    foreach(entry IN LISTS sources)
        if(entry STREQUAL "allocator.cpp" OR entry STREQUAL "${original}")
            math(EXPR matches "${matches} + 1")
            list(APPEND updated "${directory}/allocator.cpp")
        else()
            list(APPEND updated "${entry}")
        endif()
    endforeach()
    if(NOT matches EQUAL 1)
        message(FATAL_ERROR "Expected exactly one ncnn allocator compilation unit")
    endif()
    set_property(TARGET ncnn PROPERTY SOURCES "${updated}")
endfunction()
