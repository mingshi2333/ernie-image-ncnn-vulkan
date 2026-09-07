#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ernie::pipeline" for configuration "Release"
set_property(TARGET ernie::pipeline APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::pipeline PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-pipeline.a"
  )

list(APPEND _cmake_import_check_targets ernie::pipeline )
list(APPEND _cmake_import_check_files_for_ernie::pipeline "${_IMPORT_PREFIX}/lib/libernie-pipeline.a" )

# Import target "ernie::detail-runtime" for configuration "Release"
set_property(TARGET ernie::detail-runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-runtime PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-runtime.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-runtime )
list(APPEND _cmake_import_check_files_for_ernie::detail-runtime "${_IMPORT_PREFIX}/lib/libernie-runtime.a" )

# Import target "ernie::detail-pe" for configuration "Release"
set_property(TARGET ernie::detail-pe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-pe PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-pe.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-pe )
list(APPEND _cmake_import_check_files_for_ernie::detail-pe "${_IMPORT_PREFIX}/lib/libernie-pe.a" )

# Import target "ernie::detail-tokenizer" for configuration "Release"
set_property(TARGET ernie::detail-tokenizer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-tokenizer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-tokenizer.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-tokenizer )
list(APPEND _cmake_import_check_files_for_ernie::detail-tokenizer "${_IMPORT_PREFIX}/lib/libernie-tokenizer.a" )

# Import target "ernie::detail-model-package" for configuration "Release"
set_property(TARGET ernie::detail-model-package APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-model-package PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-model-package.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-model-package )
list(APPEND _cmake_import_check_files_for_ernie::detail-model-package "${_IMPORT_PREFIX}/lib/libernie-model-package.a" )

# Import target "ernie::detail-shape-graph" for configuration "Release"
set_property(TARGET ernie::detail-shape-graph APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-shape-graph PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-shape-graph.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-shape-graph )
list(APPEND _cmake_import_check_files_for_ernie::detail-shape-graph "${_IMPORT_PREFIX}/lib/libernie-shape-graph.a" )

# Import target "ernie::detail-shape-plan" for configuration "Release"
set_property(TARGET ernie::detail-shape-plan APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-shape-plan PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-shape-plan.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-shape-plan )
list(APPEND _cmake_import_check_files_for_ernie::detail-shape-plan "${_IMPORT_PREFIX}/lib/libernie-shape-plan.a" )

# Import target "ernie::detail-execution-metrics" for configuration "Release"
set_property(TARGET ernie::detail-execution-metrics APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ernie::detail-execution-metrics PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libernie-execution-metrics.a"
  )

list(APPEND _cmake_import_check_targets ernie::detail-execution-metrics )
list(APPEND _cmake_import_check_files_for_ernie::detail-execution-metrics "${_IMPORT_PREFIX}/lib/libernie-execution-metrics.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
