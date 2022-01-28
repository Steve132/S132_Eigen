
include(ExternalProject)

ExternalProject_Add(eigen_extern
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_TAG           origin/3.4
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_extern SOURCE_DIR)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE $<BUILD_INTERFACE:${SOURCE_DIR}>)




add_executable(nulltest null.cpp)

target_link_libraries(nulltest PRIVATE S132_Eigen)
target_link_libraries(nulltest PRIVATE eigen3)
