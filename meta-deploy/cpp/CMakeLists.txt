cmake_minimum_required(VERSION 3.16)
project(meta_encoder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to a built llama.cpp/ggml (override with -DLLAMA_DIR=...)
if(NOT DEFINED LLAMA_DIR)
    set(LLAMA_DIR "$ENV{HOME}/llamacpp-build")
endif()

set(GGML_INC "${LLAMA_DIR}/ggml/include")
set(GGML_LIBDIR "${LLAMA_DIR}/build/ggml/src")

find_package(Threads REQUIRED)

function(add_meta_tool name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${GGML_INC})
    target_compile_options(${name} PRIVATE -fopenmp)  # ggml is built with OpenMP → libgomp
    # The static ggml libs are interdependent → linker group.
    target_link_libraries(${name} PRIVATE
        -Wl,--start-group
        ${GGML_LIBDIR}/ggml.a
        ${GGML_LIBDIR}/ggml-cpu.a
        ${GGML_LIBDIR}/ggml-base.a
        -Wl,--end-group
        Threads::Threads
        -fopenmp
    )
endfunction()

add_meta_tool(meta_encoder)    # multi_token encoder (Gemma-2B, hardcoded sizes)
add_meta_tool(meta_ca)         # BottleneckCrossAttention
add_meta_tool(meta_selective)  # SelectiveEncoder (current; sizes from sidecar metadata)
