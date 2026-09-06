cmake_minimum_required(VERSION 3.20)
project(ErnieAllocationCliContracts LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(ERNIE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
enable_testing()
foreach(mode on off)
    add_executable(allocation-cli-${mode} ../test_allocation_cli_driver.cpp
        ${ERNIE_ROOT}/cli/options.cpp ${ERNIE_ROOT}/cli/prompt_file.cpp)
    target_include_directories(allocation-cli-${mode} PRIVATE
        ../fixtures/allocation-cli ${ERNIE_ROOT}/include ${ERNIE_ROOT}/src ${ERNIE_ROOT}/cli)
endforeach()
target_sources(allocation-cli-on PRIVATE ${ERNIE_ROOT}/cli/allocation_report.cpp
    ${ERNIE_ROOT}/src/allocation_metrics_hook.cpp ${ERNIE_ROOT}/src/execution_metrics.cpp)
target_compile_definitions(allocation-cli-on PRIVATE ERNIE_CLI_ALLOCATION_METRICS=1 ERNIE_ALLOCATION_HOOKS_ENABLED=1)
target_link_libraries(allocation-cli-on PRIVATE Threads::Threads)
add_test(NAME allocation_cli_cpu COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s ${ERNIE_ROOT}/tests -p test_allocation_cli.py -v)
set_tests_properties(allocation_cli_cpu PROPERTIES ENVIRONMENT
    "ERNIE_ALLOCATION_TEST_ON=$<TARGET_FILE:allocation-cli-on>;ERNIE_ALLOCATION_TEST_OFF=$<TARGET_FILE:allocation-cli-off>")
