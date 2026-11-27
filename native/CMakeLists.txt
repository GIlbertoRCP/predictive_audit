cmake_minimum_required(VERSION 3.24)

# Single source of truth for the version: pyproject.toml.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(_paudit_version "${SKBUILD_PROJECT_VERSION}")
else()
  file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml" _pyproject_version
       REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"")
  string(REGEX REPLACE "^version = \"([0-9.]+)\"" "\\1" _paudit_version "${_pyproject_version}")
endif()

project(paudit VERSION ${_paudit_version} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PAUDIT_BUILD_PYTHON "Build the nanobind extension module (predictive_audit._native)" OFF)
option(PAUDIT_BUILD_CLI "Build the standalone predictive-audit executable" ON)
option(PAUDIT_BUILD_TESTS "Build the Catch2 unit tests" ON)
option(PAUDIT_WERROR "Treat compiler warnings as errors" OFF)
set(PAUDIT_SANITIZE "" CACHE STRING "Comma-separated sanitizers, e.g. address,undefined")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(PauditWarnings)
include(PauditSanitizers)

# --- core library -------------------------------------------------------------
add_library(paudit STATIC
  src/version.cpp
)
add_library(paudit::paudit ALIAS paudit)
target_include_directories(paudit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)
target_compile_features(paudit PUBLIC cxx_std_20)
target_compile_definitions(paudit PRIVATE
  PAUDIT_VERSION="${PROJECT_VERSION}"
  PAUDIT_BUILD_TYPE="$<CONFIG>"
)
set_target_properties(paudit PROPERTIES POSITION_INDEPENDENT_CODE ON)
paudit_set_warnings(paudit)
paudit_enable_sanitizers(paudit)

# --- standalone CLI -----------------------------------------------------------
if(PAUDIT_BUILD_CLI)
  add_executable(paudit_cli cli/main.cpp)
  set_target_properties(paudit_cli PROPERTIES OUTPUT_NAME predictive-audit)
  target_link_libraries(paudit_cli PRIVATE paudit::paudit)
  paudit_set_warnings(paudit_cli)
  paudit_enable_sanitizers(paudit_cli)
endif()

# --- Python extension ---------------------------------------------------------
if(PAUDIT_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_native NB_STATIC python/module.cpp)
  target_link_libraries(_native PRIVATE paudit::paudit)
  paudit_set_warnings(_native)
  install(TARGETS _native LIBRARY DESTINATION predictive_audit)
endif()

# --- tests --------------------------------------------------------------------
if(PAUDIT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    SYSTEM
  )
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_executable(paudit_tests tests/test_version.cpp)
  target_link_libraries(paudit_tests PRIVATE paudit::paudit Catch2::Catch2WithMain)
  paudit_set_warnings(paudit_tests)
  paudit_enable_sanitizers(paudit_tests)
  include(Catch)
  catch_discover_tests(paudit_tests)
endif()
