cmake_minimum_required(VERSION 3.20)

set(GOOGLETEST_VERSION 1.9.0)

option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)

add_subdirectory(googletest)
