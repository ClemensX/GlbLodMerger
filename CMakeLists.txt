cmake_minimum_required(VERSION 4.0)
project(glbLodMerger LANGUAGES CXX)

# Sources
set(SOURCES
    glbLodMerger.cpp
    glbLodMerger.h
)

add_executable(glbLodMerger ${SOURCES})

# C++ standard and warnings
set_target_properties(glbLodMerger PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(glbLodMerger PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
    target_compile_definitions(glbLodMerger PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(glbLodMerger PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Define TinyGLTF implementation only in the single TU that includes tiny_gltf.h
# This avoids ODR violations if you add more sources later.
set_source_files_properties(src/app/glbLodMerger.cpp
    PROPERTIES COMPILE_DEFINITIONS "TINYGLTF_IMPLEMENTATION"
)

# Dependencies
# 1) GLM: prefer a package (e.g., via vcpkg) and fallback to header-only include path
# Try common config package names
set(HAVE_GLM FALSE)
find_package(glm CONFIG QUIET)
if(TARGET glm::glm)
    target_link_libraries(glbLodMerger PRIVATE glm::glm)
    set(HAVE_GLM TRUE)
endif()

# Optional: user-provided GLM include directory if no package was found
if(NOT HAVE_GLM)
    # Set this from the command line if needed:
    #   -DGLM_INCLUDE_DIR="C:/path/to/glm"
    if(DEFINED GLM_INCLUDE_DIR)
        target_include_directories(glbLodMerger PRIVATE "${GLM_INCLUDE_DIR}")
    else()
        message(STATUS "GLM package not found. Define GLM_INCLUDE_DIR to the folder that contains glm headers (the folder that contains 'glm/').")
    endif()
endif()

find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
target_include_directories(glbLodMerger PRIVATE ${TINYGLTF_INCLUDE_DIRS})

# Nice defaults for Visual Studio debugger
if(MSVC)
    # Set working directory to the solution root when debugging
    set_property(TARGET glbLodMerger PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# Optional: install rule
install(TARGETS glbLodMerger RUNTIME DESTINATION bin)