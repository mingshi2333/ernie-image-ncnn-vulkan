# An installed application sees only ernie::pipeline and the standard C++ API.
# Static implementation archives retain link dependencies but no private headers.
include(CMakePackageConfigHelpers)
set_target_properties(ernie-pipeline PROPERTIES EXPORT_NAME pipeline)
foreach(component runtime pe tokenizer model-package shape-graph)
    set_target_properties(ernie-${component} PROPERTIES EXPORT_NAME detail-${component})
endforeach()
install(TARGETS ernie-pipeline ernie-runtime ernie-pe ernie-tokenizer
                ernie-model-package ernie-shape-graph
    EXPORT ErnieTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "$<TARGET_FILE:ernie-tokenizer-native>" DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${PROJECT_SOURCE_DIR}/include/ernie/pipeline.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ernie)
install(EXPORT ErnieTargets NAMESPACE ernie:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ernie)

get_filename_component(ERNIE_TOKENIZER_ARCHIVE "${tokenizer_library}" NAME)
if(NOT ERNIE_TOKENIZER_ARCHIVE)
    get_target_property(ernie_tokenizer_path ernie-tokenizer-native IMPORTED_LOCATION)
    get_filename_component(ERNIE_TOKENIZER_ARCHIVE "${ernie_tokenizer_path}" NAME)
endif()
configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/ErnieConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ErnieConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ernie
    PATH_VARS CMAKE_INSTALL_LIBDIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ErnieConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ErnieConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ErnieConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ernie)
