add_executable(nserun
                main.cpp
                initializeFlow.cpp
                MyFunctions.H)

# uncomment below line for gpu compilation
# setup_target_for_cuda_compilation(nserun)

target_include_directories(nserun PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Link to the SAME LGF_core_lib to use MeshData and kernels
target_link_libraries(nserun PRIVATE LGF_core_lib NSE_core_lib AMReX::amrex)

set_target_properties(nserun PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test"
)

# Update the input file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/inputs"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/inputs"
    COMMENT "Syncing inputs to Exec/NSE_test..."
)

# Update the jobscript file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/debug.sh"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/debug.sh"
    COMMENT "Syncing jobscript to Exec/NSE_test..."
)

# Update the postprocessing file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/postProcess.py"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/postProcess.py"
    COMMENT "Syncing postprocessing script to Exec/NSE_test..."
)