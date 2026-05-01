file(GLOB CUDA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach(CUDA_FILE ${CUDA_FILES})
    get_filename_component(EXEC_NAME ${CUDA_FILE} NAME_WE)

    add_executable(${EXEC_NAME} ${CUDA_FILE})

    target_link_libraries(${EXEC_NAME}
        PRIVATE
        CUDA::cudart
        CUDA::cublas
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${EXEC_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
endforeach()