find_package(PNG REQUIRED)
add_executable(ernie-image main.cpp options.cpp prompt_file.cpp image_io.cpp)
target_include_directories(ernie-image PRIVATE "${PROJECT_SOURCE_DIR}/third_party/stb")
target_link_libraries(ernie-image PRIVATE ernie::pipeline PNG::PNG)
include(GNUInstallDirs)
install(TARGETS ernie-image RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/sources.lock.json"
              "${PROJECT_SOURCE_DIR}/docs/RUNNING.md" DESTINATION ${CMAKE_INSTALL_DATADIR}/ernie-image)
