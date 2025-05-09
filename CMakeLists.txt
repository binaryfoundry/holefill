cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME HoleFillingCLI)

# Set project name and version
project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the startup project for Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Add STB image library
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_compile_definitions(stb INTERFACE
    STB_IMAGE_IMPLEMENTATION
    STB_IMAGE_WRITE_IMPLEMENTATION
)

# Add nanoflann library
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_compile_definitions(nanoflann INTERFACE
    NANOFLANN_USE_OPENMP
)

# Add source and header files for the library
set(HOLEFILL_SOURCES
    src/holefill.cpp)

set(HOLEFILL_HEADERS
    src/holefill.h)

# Create the static library
add_library(holefill STATIC ${HOLEFILL_SOURCES} ${HOLEFILL_HEADERS})

# Add main executable source
set(MAIN_SOURCE
    src/main.cpp)

# Define the executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCE})

# Link the static library to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE holefill stb nanoflann)
target_link_libraries(holefill PRIVATE stb nanoflann)

# Enable the use of folders in Visual Studio and some other IDEs that support CMake-generated project files.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Group headers and source files in Visual Studio's Solution Explorer
source_group("Source Files" FILES ${MAIN_SOURCE})
source_group("Header Files" FILES ${HOLEFILL_HEADERS})

# Set compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
    target_compile_options(holefill PRIVATE /W4 /permissive-)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(holefill PRIVATE -Wall -Wextra -Wpedantic)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Define build types explicitly
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Copy resources directory to output directory
if(MSVC)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/
    )
else()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    )
endif()
