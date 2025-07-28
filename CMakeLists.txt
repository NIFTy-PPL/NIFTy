cmake_minimum_required(VERSION 3.18)
project(nifty_cl_cpp LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)


execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_nifty_cl_core
    nifty/cl_cpp/bindings.cpp
)

set_target_properties(_nifty_cl_core PROPERTIES
    OUTPUT_NAME "_nifty_cl_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nifty/cl_cpp"
)

install(TARGETS _nifty_cl_core LIBRARY DESTINATION nifty/cl_cpp)
