cmake_minimum_required(VERSION 3.18.0)
project(BCSKMutualInformation LANGUAGES CXX CUDA)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set stack size for C++
set(CMAKE_CXX_STACK_SIZE "1000000000")

# Set HAVE_OPENMP based on WITH_OPENMP
set(HAVE_OPENMP ${WITH_OPENMP})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDA)

# Compiler-specific configurations
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(-DCRT_SECURE_NO_WARNINGS -DLAMBDAEXPRESSIONS -DUNICODE -DQT_LARGEFILE_SUPPORT -DNOMINMAX)
    
    if(HAVE_OPENMP)
        set(FLAGS_OPENMP "/openmp")
    endif()

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP /Ob1 /Oy- /GF ${FLAGS_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP /Ox /Ob2 /Oi /Ot /Oy /GF /Gm- /GS- /Gy /fp:fast ${FLAGS_OPENMP}")

    if(MSVC32)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:SSE2")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function -Werror=return-type -march=native -Wa,-q")

    if(WITH_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wa,-q -fabi-version=0")
    endif()

    if(WITH_INSPECTION)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()

    if(HAVE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()


# Add CUDA executable
set(SOURCES
    main.cpp
)

# CUDA-specific configurations
if(CUDA_FOUND)
    enable_language(CUDA)

    # Automatically detect and set CUDA architecture
    include(FindCUDA/select_compute_arch)
    CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_ARCH_FLAGS)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_ARCH_FLAGS})

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARIES})

    set(SOURCES ${SOURCES} kernels.cu)

    add_definitions(-DENABLE_CUDA)
endif()


include_directories(include)
include_directories(3rdparty/include)

add_executable(bcsk_mutinf ${SOURCES})

if(CUDA_FOUND)
    target_link_libraries(bcsk_mutinf ${CUDA_LIBRARIES})
endif()

# Link OpenMP library
if(OpenMP_CXX_FOUND)
    target_link_libraries(bcsk_mutinf OpenMP::OpenMP_CXX)
endif()
