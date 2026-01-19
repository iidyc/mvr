cmake_minimum_required(VERSION 3.24)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(CMAKE_CXX_FLAGS  "-lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")

include_directories("/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/include")

include_directories("RaBitQ-Library/include")

# add_subdirectory(RaBitQ-Library)

# add_executable(my_app main.cpp)
    
# target_link_libraries(my_app
#     PRIVATE
#         "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
# )

add_executable(search search.cpp)
    
target_link_libraries(search
    PRIVATE
        "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
)

add_executable(build_ivf build_IVFRaBitQ.cpp)
    
target_link_libraries(build_ivf
    PRIVATE
        "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
)

add_executable(search_exbits search_exbits.cpp)
    
target_link_libraries(search_exbits
    PRIVATE
        "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
)

add_executable(rabitqex_ivf_est expr/rabitqex_ivf_est.cpp)

target_link_libraries(rabitqex_ivf_est
    PRIVATE
        "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
)

add_executable(rabitqex_ivf_gather_est expr/rabitqex_ivf_gather_est.cpp)

target_link_libraries(rabitqex_ivf_gather_est
    PRIVATE
        "/work/pi_ameliou_umass_edu/yanqichen/.conda/envs/faiss/lib/libfaiss_avx2.so"
)
