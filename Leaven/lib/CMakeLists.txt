cmake_minimum_required(VERSION 3.10)
project(LeavenLib)

set(CMAKE_CXX_STANDARD 14)

# DiscreGrid
include_directories(ext/Discregrid/discregrid/include)
include_directories(ext/Discregrid/discregrid/include/Discregrid)
file(GLOB_RECURSE GRID_HEADERS ext/Discregrid/discregrid/include/*.h ext/Discregrid/discregrid/include/*.hpp ext/Discregrid/discregrid/include/*.inl ext/Discregrid/discregrid/include/All)
file(GLOB_RECURSE GRID_SOURCES ext/Discregrid/discregrid/src/*.cpp)

# Get the EIGEN environment variable. Since EIGEN is a header-only library, we
# just need to add it to the include directory.
SET(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
IF(NOT EIGEN3_INCLUDE_DIR)
	# The environment variable was not set
	SET(ERR_MSG "Please point the environment variable EIGEN3_INCLUDE_DIR to the root directory of your EIGEN installation.")
	IF(WIN32)
		# On Windows, try the default location
		MESSAGE(STATUS "Looking for EIGEN in ${DEF_DIR_EIGEN}")
		IF(IS_DIRECTORY ${DEF_DIR_EIGEN})
			MESSAGE(STATUS "Found!")
			SET(EIGEN3_INCLUDE_DIR ${DEF_DIR_EIGEN})
		ELSE()
			MESSAGE(FATAL_ERROR ${ERR_MSG})
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR ${ERR_MSG})
	ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# Find all Eigen3 an4 Source files
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS src/*.h)
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES src/*.cpp)

add_library(${PROJECT_NAME}
        ${${PROJECT_NAME}_SOURCES}
        ${${PROJECT_NAME}_HEADERS}
        ${GRID_HEADERS}
        ${GRID_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PUBLIC ext/Discregrid/discregrid/include)
