cmake_minimum_required(VERSION 3.15)
project(HGS_CVRP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI)

set(
    src_files
    Program/CudaSanity.cu
    Program/Genetic.cpp
    Program/Individual.cpp
    Program/LocalSearch.cpp
    Program/Params.cpp
    Program/Population.cpp
    Program/Split.cpp
    Program/InstanceCVRPLIB.cpp
    Program/AlgorithmParameters.cpp
    Program/C_Interface.cpp
    Program/IslandModels/IslandCommunicators/MPIIslandCommunicator.cpp
)

include_directories(Program Program/IslandModels Program/IslandModels/ImmigrantHandlers Program/IslandModels/IslandCommunicators Program/IslandModels/MigrantSelectors Program/IslandModels/MigrationPolicies Program/IslandModels/Topologies)

add_executable(bin Program/main.cpp ${src_files})
set_target_properties(bin PROPERTIES
    OUTPUT_NAME hgs
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(bin PRIVATE CUDA::cudart)


if(MPI_FOUND)
    target_link_libraries(bin PRIVATE MPI::MPI_CXX)
    target_compile_definitions(bin PRIVATE USE_MPI)
    message(STATUS "MPI found")
else()
    message(STATUS "MPI not found")
endif()