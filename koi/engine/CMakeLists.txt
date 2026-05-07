cmake_minimum_required(VERSION 3.18)
project(mini_engine C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# strict flags — INFER spec requires -Wall -Wextra -Werror
add_compile_options(-Wall -Wextra -Werror -O2)

# static library
add_library(mini STATIC
    src/loader.c
    src/tokenizer.c
    src/kernels_scalar.c
    src/model.c
    src/sample.c
    src/mini.c
)
target_include_directories(mini PUBLIC include)
target_include_directories(mini PRIVATE src)
target_link_libraries(mini m)  # libm for sqrtf/expf/cosf/sinf

# CLI binary
add_executable(mini_cli cli/main.c)
target_include_directories(mini_cli PRIVATE include)
target_link_libraries(mini_cli mini)
set_target_properties(mini_cli PROPERTIES OUTPUT_NAME "mini")

# tests (placeholder — golden-output tests added at M1 completion)
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# install
install(TARGETS mini_cli DESTINATION bin)
install(FILES include/mini.h DESTINATION include)
