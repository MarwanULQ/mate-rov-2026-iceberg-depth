cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME zed_open_capture)

project(${PROJECT_NAME} CXX)

############################################################################
# Flags

cmake_policy(SET CMP0054 NEW)

# if CMAKE_BUILD_TYPE is not specified, take 'Release' as default
set(${PROJECT_NAME}_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS "${${PROJECT_NAME}_CXX_FLAGS} -Wall -pedantic -g")

if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -flto") # File size optimization
    message("* Release mode")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("* Debug mode")
    SET(CMAKE_DEBUG_POSTFIX  "d")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -flto") # File size optimization
    message("* Release mode")
endif()

# Allows to start application by double clicking the icon on Ubuntu
set ( CMAKE_CXX_FLAGS "-no-pie" )

# CPU information
message("Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# Raspberry Pi 4 - OS 64bit
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("* Optimizing for ${CMAKE_SYSTEM_PROCESSOR}")
  add_compile_options(-mtune=cortex-a72 -mcpu=cortex-a72)
  add_definitions(-DEMBEDDED_ARM)
endif()

# Raspberry Pi 4 - OS 32bit
if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
  message("* Optimizing for ${CMAKE_SYSTEM_PROCESSOR}")
  # add_compile_options(-mcpu=cortex-a72 -mfloat-abi=hard -mfpu=neon-fp-armv8 -mneon-for-64bits -mtune=cortex-a72)
  add_compile_options(-mfloat-abi=hard -mfpu=neon-fp-armv8 -mneon-for-64bits -mtune=cortex-a72)
  add_definitions(-DEMBEDDED_ARM)
endif()


############################################################################
# Options
option(BUILD_VIDEO      "Build the ZED Open Capture Video Modules (only for Linux)"   ON)
option(BUILD_SENSORS    "Build the ZED Open Capture Sensors Modules"                  ON)
option(BUILD_EXAMPLES   "Build the ZED Open Capture examples"                         ON)
option(DEBUG_CAM_REG    "Add functions to log the values of the registers of camera"  OFF)
option(BUILD_TORCH_EXAMPLE "Build the TorchScript measurement example (measure_distance_v4)" ON)
set(LIBTORCH_ROOT "" CACHE PATH "Path to a local LibTorch installation root")

set(DEFAULT_LIBTORCH_ROOT "${PROJECT_SOURCE_DIR}/../../third_party/libtorch")
if(NOT LIBTORCH_ROOT AND EXISTS "${DEFAULT_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    set(LIBTORCH_ROOT "${DEFAULT_LIBTORCH_ROOT}")
endif()

# Torch's CMake modules may enable CUDA; avoid empty-architecture cache failures.
if(BUILD_TORCH_EXAMPLE)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
    endif()
endif()

############################################################################
# Sources
set(SRC_VIDEO
    ${PROJECT_SOURCE_DIR}/src/videocapture.cpp
)

set(SRC_SENSORS
    ${PROJECT_SOURCE_DIR}/src/sensorcapture.cpp
)

############################################################################
# Includes
set(HEADERS_VIDEO
    # Base
    ${PROJECT_SOURCE_DIR}/include/videocapture.hpp
    
    # Defines
    ${PROJECT_SOURCE_DIR}/include/defines.hpp
    ${PROJECT_SOURCE_DIR}/include/videocapture_def.hpp
)

set(HEADERS_SENSORS
    # Base
    ${PROJECT_SOURCE_DIR}/include/sensorcapture.hpp

    # Defines
    ${PROJECT_SOURCE_DIR}/include/defines.hpp
    ${PROJECT_SOURCE_DIR}/include/sensorcapture_def.hpp
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

############################################################################
# Required external libraries

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")

############################################################################
# Generate libraries
if(DEBUG_CAM_REG)
    message("* Registers logging available")
    add_definitions(-DSENSOR_LOG_AVAILABLE)
endif()

if(BUILD_SENSORS)
    message("* Sensors module available")
    add_definitions(-DSENSORS_MOD_AVAILABLE)

    find_package(HIDAPI REQUIRED)
    message(STATUS "libhidapi: include dir at ${HIDAPI_INCLUDE_DIRS}")
    message(STATUS "libhidapi: libraries at ${HIDAPI_LIBRARIES}")
    include_directories(${HIDAPI_INCLUDE_DIRS})

    set(SRC_FULL ${SRC_FULL} ${SRC_SENSORS})
    set(HDR_FULL ${HDR_FULL} ${HEADERS_SENSORS})
    set(DEP_LIBS ${DEP_LIBS}
        ${LibUSB_LIBRARIES}
        ${HIDAPI_LIBRARIES}
        pthread )
endif()

if(BUILD_VIDEO)
    message("* Video module available")
    add_definitions(-DVIDEO_MOD_AVAILABLE)

    find_package(LIBUSB 1 REQUIRED)
    message(STATUS "libusb: include dir at ${LibUSB_INCLUDE_DIRS}")
    message(STATUS "libusb: libraries at ${LibUSB_LIBRARIES}") 
    include_directories(${LibUSB_INCLUDE_DIRS})
    set(SRC_FULL ${SRC_FULL} ${SRC_VIDEO})
    set(HDR_FULL ${HDR_FULL} ${HEADERS_VIDEO})
    set(DEP_LIBS ${DEP_LIBS}
         ${LibUSB_LIBRARIES}
         )

endif()

add_library(${PROJECT_NAME} SHARED ${SRC_FULL} )
target_link_libraries( ${PROJECT_NAME}  ${DEP_LIBS})

add_library(cnpy
    src/third_party/cnpy/cnpy.cpp
)

target_include_directories(cnpy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/cnpy
)

target_link_libraries(cnpy z)

############################################################################
# Install rules
set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HDR_FULL}"
)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/zed-open-capture)


# Generate examples
if(BUILD_EXAMPLES)
    if(BUILD_VIDEO)
        message("* Selected video examples available (v3 and v4)")

        ##### OpenCV is required
        find_package(OpenCV REQUIRED
            COMPONENTS core imgproc highgui calib3d ximgproc
            HINTS /usr/lib/x86_64-linux-gnu/cmake/opencv4
        )
        include_directories(${OpenCV_INCLUDE_DIRS})
        include_directories(${PROJECT_SOURCE_DIR}/examples/include)

        # ##### measure_distance_v1 Example (disabled: release supports v3/v4 only)
        # set(MEASURE_DISTANCE_EXAMPLE ${PROJECT_NAME}_measure_distance_v1)
        # add_executable(${MEASURE_DISTANCE_EXAMPLE}
        #     "${PROJECT_SOURCE_DIR}/examples/measure_distance_v1.cpp")
        # set_target_properties(${MEASURE_DISTANCE_EXAMPLE} PROPERTIES PREFIX "")
        # set_property(TARGET ${MEASURE_DISTANCE_EXAMPLE}
        #     PROPERTY POSITION_INDEPENDENT_CODE ON)
        # target_link_libraries(${MEASURE_DISTANCE_EXAMPLE}
        #     ${PROJECT_NAME}
        #     ${OpenCV_LIBS}
        #     cnpy
        # )
        # install(TARGETS ${MEASURE_DISTANCE_EXAMPLE}
        #     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        # )

        ##### measure_distance_v3 Example (high-quality stereo pipeline)
        set(MEASURE_DISTANCE_V3_EXAMPLE ${PROJECT_NAME}_measure_distance_v3)
        add_executable(${MEASURE_DISTANCE_V3_EXAMPLE}
            "${PROJECT_SOURCE_DIR}/examples/measure_distance_v3.cpp")
        set_target_properties(${MEASURE_DISTANCE_V3_EXAMPLE} PROPERTIES PREFIX "")
        set_property(TARGET ${MEASURE_DISTANCE_V3_EXAMPLE}
            PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${MEASURE_DISTANCE_V3_EXAMPLE}
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            cnpy
        )
        install(TARGETS ${MEASURE_DISTANCE_V3_EXAMPLE}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        )

        if(BUILD_TORCH_EXAMPLE)
            find_package(nlohmann_json QUIET)

            if(Torch_DIR)
                message(STATUS "Using Torch_DIR=${Torch_DIR}")
                find_package(Torch QUIET)
            elseif(LIBTORCH_ROOT)
                file(TO_CMAKE_PATH "${LIBTORCH_ROOT}" LIBTORCH_ROOT_CMAKE)
                set(Torch_DIR "${LIBTORCH_ROOT_CMAKE}/share/cmake/Torch")
                message(STATUS "Using LibTorch root at ${LIBTORCH_ROOT_CMAKE}")
                find_package(Torch QUIET)
            endif()

            if(Torch_FOUND AND nlohmann_json_FOUND)
                message("* Torch measurement example available")

                # set(MEASURE_DISTANCE_V2_EXAMPLE ${PROJECT_NAME}_measure_distance_v2)
                # add_executable(${MEASURE_DISTANCE_V2_EXAMPLE}
                #     "${PROJECT_SOURCE_DIR}/examples/measure_distance_v2.cpp")
                # set_target_properties(${MEASURE_DISTANCE_V2_EXAMPLE} PROPERTIES
                #     PREFIX ""
                #     CXX_STANDARD 17
                #     CXX_STANDARD_REQUIRED ON
                # )

                # target_compile_options(${MEASURE_DISTANCE_V2_EXAMPLE} PRIVATE ${TORCH_CXX_FLAGS})

                # target_link_libraries(${MEASURE_DISTANCE_V2_EXAMPLE}
                #     ${PROJECT_NAME}
                #     ${OpenCV_LIBS}
                #     cnpy
                #     ${TORCH_LIBRARIES}
                #     nlohmann_json::nlohmann_json
                # )

                # install(TARGETS ${MEASURE_DISTANCE_V2_EXAMPLE}
                #     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                # )

                set(MEASURE_DISTANCE_V4_EXAMPLE ${PROJECT_NAME}_measure_distance_v4)
                add_executable(${MEASURE_DISTANCE_V4_EXAMPLE}
                    "${PROJECT_SOURCE_DIR}/examples/measure_distance_v4.cpp")
                set_target_properties(${MEASURE_DISTANCE_V4_EXAMPLE} PROPERTIES
                    PREFIX ""
                    CXX_STANDARD 17
                    CXX_STANDARD_REQUIRED ON
                )

                target_compile_options(${MEASURE_DISTANCE_V4_EXAMPLE} PRIVATE ${TORCH_CXX_FLAGS})

                target_link_libraries(${MEASURE_DISTANCE_V4_EXAMPLE}
                    ${PROJECT_NAME}
                    ${OpenCV_LIBS}
                    cnpy
                    ${TORCH_LIBRARIES}
                    nlohmann_json::nlohmann_json
                )

                install(TARGETS ${MEASURE_DISTANCE_V4_EXAMPLE}
                    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                )
            else()
                if(NOT Torch_FOUND)
                    message(WARNING "BUILD_TORCH_EXAMPLE=ON but LibTorch was not found. Set -DLIBTORCH_ROOT=/path/to/libtorch or -DTorch_DIR=/path/to/libtorch/share/cmake/Torch.")
                endif()

                if(NOT nlohmann_json_FOUND)
                    message(WARNING "BUILD_TORCH_EXAMPLE=ON but nlohmann_json was not found. Skipping measure_distance_v4 target.")
                endif()
            endif()
        endif()

        # ##### Depth Example (disabled: release supports v3/v4 only)
        # find_package(OpenMP)

        # set(DEPTH_EXAMPLE ${PROJECT_NAME}_depth_example)
        # include_directories(${PROJECT_SOURCE_DIR}/examples/include)
        # add_executable(${DEPTH_EXAMPLE}
        #     "${PROJECT_SOURCE_DIR}/examples/zed_oc_depth_example.cpp")
        # set_target_properties(${DEPTH_EXAMPLE} PROPERTIES PREFIX "")

        # if(OpenMP_CXX_FOUND)
        #     include_directories(${OpenMP_CXX_INCLUDE_DIRS})
        #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        #     target_link_libraries(${DEPTH_EXAMPLE}
        #         OpenMP::OpenMP_CXX
        #         ${PROJECT_NAME}
        #         ${OpenCV_LIBS}
        #     )
        # else()
        #     target_link_libraries(${DEPTH_EXAMPLE}
        #         ${PROJECT_NAME}
        #         ${OpenCV_LIBS}
        #     )
        # endif()

        # --------------------------------------------------------------------
        # Disabled examples
        # --------------------------------------------------------------------

        # ##### Video Example
        # set(VIDEO_EXAMPLE ${PROJECT_NAME}_video_example)
        # add_executable(${VIDEO_EXAMPLE}
        #     "${PROJECT_SOURCE_DIR}/examples/zed_oc_video_example.cpp")
        # set_target_properties(${VIDEO_EXAMPLE} PROPERTIES PREFIX "")
        # set_property(TARGET ${VIDEO_EXAMPLE}
        #     PROPERTY POSITION_INDEPENDENT_CODE ON)
        # target_link_libraries(${VIDEO_EXAMPLE}
        #     ${PROJECT_NAME}
        #     ${OpenCV_LIBS}
        # )

        # ##### Control Example
        # set(CONTROL_EXAMPLE ${PROJECT_NAME}_control_example)
        # add_executable(${CONTROL_EXAMPLE}
        #     "${PROJECT_SOURCE_DIR}/examples/zed_oc_control_example.cpp")
        # set_target_properties(${CONTROL_EXAMPLE} PROPERTIES PREFIX "")
        # target_link_libraries(${CONTROL_EXAMPLE}
        #     ${PROJECT_NAME}
        #     ${OpenCV_LIBS}
        # )

        # ##### Rectify Example
        # set(RECTIFY_EXAMPLE ${PROJECT_NAME}_rectify_example)
        # include_directories(${PROJECT_SOURCE_DIR}/examples/include)
        # add_executable(${RECTIFY_EXAMPLE}
        #     "${PROJECT_SOURCE_DIR}/examples/zed_oc_rectify_example.cpp")
        # set_target_properties(${RECTIFY_EXAMPLE} PROPERTIES PREFIX "")
        # target_link_libraries(${RECTIFY_EXAMPLE}
        #     ${PROJECT_NAME}
        #     ${OpenCV_LIBS}
        # )

        # ##### Stereo Tune App (disabled: release supports v3/v4 only)
        # set(STEREO_TUNE_APP ${PROJECT_NAME}_depth_tune_stereo)
        # include_directories(${PROJECT_SOURCE_DIR}/examples/include)
        # add_executable(${STEREO_TUNE_APP}
        #     "${PROJECT_SOURCE_DIR}/examples/tools/zed_oc_tune_stereo_sgbm.cpp")
        # set_target_properties(${STEREO_TUNE_APP} PROPERTIES PREFIX "")
        # target_link_libraries(${STEREO_TUNE_APP}
        #     ${PROJECT_NAME}
        #     ${OpenCV_LIBS}
        #     cnpy
        # )

        # if(DEBUG_CAM_REG)
        #     add_executable(${PROJECT_NAME}_video_reg_log
        #         "${PROJECT_SOURCE_DIR}/examples/zed_oc_video_reg_log.cpp")
        #     set_target_properties(${PROJECT_NAME}_video_reg_log PROPERTIES PREFIX "")
        #     target_link_libraries(${PROJECT_NAME}_video_reg_log
        #         ${PROJECT_NAME}
        #         ${OpenCV_LIBS}
        #     )
        # endif()
    endif()

    # if(BUILD_SENSORS)
    #     message("* Sensors example available")
    #
    #     add_executable(${PROJECT_NAME}_sensors_example
    #         "${PROJECT_SOURCE_DIR}/examples/zed_oc_sensors_example.cpp")
    #     set_target_properties(${PROJECT_NAME}_sensors_example PROPERTIES PREFIX "")
    #     target_link_libraries(${PROJECT_NAME}_sensors_example
    #         ${PROJECT_NAME}
    #     )
    # endif()

    # if(BUILD_VIDEO AND BUILD_SENSORS)
    #     message("* Video/Sensors sync example available")
    #
    #     add_executable(${PROJECT_NAME}_sync_example
    #         "${PROJECT_SOURCE_DIR}/examples/zed_oc_sync_example.cpp")
    #     set_target_properties(${PROJECT_NAME}_sync_example PROPERTIES PREFIX "")
    #     target_link_libraries(${PROJECT_NAME}_sync_example
    #         ${PROJECT_NAME}
    #         ${OpenCV_LIBS}
    #     )
    # endif()
endif()