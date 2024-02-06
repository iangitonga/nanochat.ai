cmake_minimum_required(VERSION 3.25)

# Name of the project (will be the name of the plugin)
project (backend)

set(CMAKE_CXX_STANDARD 17)
# Don't add this line if you will try_compile with boost.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Check for the presence of AVX and figure out the flags to use for it.
macro(CHECK_FOR_AVX)
    set(AVX_FLAGS)

    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)

    # Check for AVX
    if (MSVC)
        if (NOT MSVC_VERSION LESS 1600)
            set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
        endif ()
    else ()
        set(CMAKE_REQUIRED_FLAGS "-mavx")
    endif ()

    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m256 a, b, c;
          const float src[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
          float dst[8];
          a = _mm256_loadu_ps( src );
          b = _mm256_loadu_ps( src );
          c = _mm256_add_ps( a, b );
          _mm256_storeu_ps( dst, c );
          for( int i = 0; i < 8; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }
          return 0;
        }"
            HAVE_AVX_EXTENSIONS)

    # Set Flags according to check results
    if (MSVC)
        if (HAVE_AVX_EXTENSIONS AND NOT MSVC_VERSION LESS 1600)
            set(AVX_FLAGS "${AVX_FLAGS} /arch:AVX")
        endif ()
    else ()
        if (HAVE_AVX_EXTENSIONS)
            set(AVX_FLAGS "${AVX_FLAGS}-mavx")
            message("LOG: AVX detected")
        endif ()
    endif ()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAGS}")
endmacro(CHECK_FOR_AVX)


CHECK_FOR_AVX()

# Essential include files to build a node addon,
# you should add this line in every CMake.js based project.
include_directories(${CMAKE_JS_INC})

# Declare the location of the source files
# file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h")
file(GLOB SOURCE_FILES "src/backend/backend.cpp")

# This line will tell CMake that we're building a shared library
# from the above source files
# named after the project's name
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# This line will give our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential library files to link to a node addon,
# you should add this line in every CMake.js based project.
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

# Define NAPI_VERSION
add_definitions(-DNAPI_VERSION=8)