
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ErnieConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
set(Ernie_NCNN_REVISION "6a1bf000f363714839a36793addc8c879d3d899e")
set(Ernie_VULKAN_ENABLED OFF)
if(NOT TARGET ernie::pipeline)
    # A separately imported ncnn may have different layers, ABI or mathematics.
    # The static SDK must link the copy installed in this same prefix.
    if(TARGET ncnn)
        set(Ernie_FOUND FALSE)
        set(Ernie_NOT_FOUND_MESSAGE "Load Ernie before any separate ncnn package; Ernie uses its bundled pinned ncnn")
        return()
    endif()
    set_and_check(_ernie_libdir "${PACKAGE_PREFIX_DIR}/lib")
    find_dependency(Threads)
    # find_package still prefers a cached ncnn_DIR even with NO_DEFAULT_PATH.
    # Include this exact installed config without changing the caller's cache.
    set_and_check(_ernie_ncnn_config "${_ernie_libdir}/cmake/ncnn/ncnnConfig.cmake")
    include("${_ernie_ncnn_config}")
    set_and_check(_ernie_tokenizer "${_ernie_libdir}/libernie_tokenizer_bridge.a")
    add_library(ernie::tokenizer-native STATIC IMPORTED)
    set_target_properties(ernie::tokenizer-native PROPERTIES IMPORTED_LOCATION "${_ernie_tokenizer}")
    include("${CMAKE_CURRENT_LIST_DIR}/ErnieTargets.cmake")
    unset(_ernie_tokenizer)
    unset(_ernie_ncnn_config)
    unset(_ernie_libdir)
endif()
check_required_components(Ernie)
