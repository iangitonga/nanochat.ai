cmake_minimum_required(VERSION 3.0)

# Name of the project (will be the name of the plugin)
project (backend)

set(CMAKE_CXX_STANDARD 17)
# Don't add this line if you will try_compile with boost.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
endif()

# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    message("GTEN LOG: OpenMP detected")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Check for the presence of AVX and figure out the flags to use for it.
macro(CHECK_FOR_AVX)
    set(AVX_FLAGS)

    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)

    # Set the flags for compilation of the source below.
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
            message("GTEN LOG: AVX detected")
        endif ()
    endif ()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAGS}")
endmacro(CHECK_FOR_AVX)


# Check for the presence of mf16c and figure out the flags to use for it.
# mf16c allows loading and writing fp16 values and converting them in-hardware.
# TODO: Does AVX detection imply f16? in that case no need to check for it here!
macro(CHECK_FOR_F16C)
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)

    # Set the flags for compilation of the source below.
    if (NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "-mavx -mf16c")
    endif ()

    check_cxx_source_runs("
        #include <immintrin.h>
        #include <cstdint>
        
        int main()
        {
            const float flts[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            //                  1.0    2.0    3.0    4.0    5.0    6.0    7.0    8.0
            const uint16_t halfs[9] = {15360, 16384, 16896, 17408, 17664, 17920, 18176, 18432};
            const __m256 cvt_flts = _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)halfs));
            const float* cvt_flts_ptr = (float*)&cvt_flts;
            for (int i = 0; i < 8; ++i) {
                if (cvt_flts_ptr[i] != flts[i]) {
                    return -1;
                } 
            }
            uint16_t cvt_halfs[8];
            _mm_storeu_si128((__m128i_u *)cvt_halfs, _mm256_cvtps_ph(cvt_flts, 0));
            for (int i = 0; i < 8; ++i) {
                if (cvt_halfs[i] != halfs[i]) {
                    return -1;
                } 
            }
            return 0;
        }"
            HAVE_FPC_EXTENSIONS)

    # Set Flags according to check results
    # TODO: support MSVC.
    if (NOT MSVC)
        if (HAVE_FPC_EXTENSIONS)
            message("GTEN LOG: F16C detected")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c")
        endif()
    endif ()
endmacro(CHECK_FOR_F16C)


# Include AVX if available.
CHECK_FOR_AVX()
CHECK_FOR_F16C()

# Essential include files to build a node addon,
# you should add this line in every CMake.js based project.
include_directories(${CMAKE_JS_INC})

# Declare the location of the source files
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/backend/*.cpp")

# This line will tell CMake that we're building a shared library
# from the above source files
# named after the project's name
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# This line will give our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# nanochat include
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/src/backend/")

# Essential library files to link to a node addon,
# you should add this line in every CMake.js based project.
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

# Define NAPI_VERSION
add_definitions(-DNAPI_VERSION=8)