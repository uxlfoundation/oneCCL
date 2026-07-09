# OCCP (OneCCL Coordinator Protocol) Project Makefile
# =============================================================================

# Project Configuration
PROJECT_NAME := occp
VERSION := 1.0.0

# Directories
SRC_DIR := ..
BUILD_DIR := build
DEBUG_DIR := $(BUILD_DIR)/debug
RELEASE_DIR := $(BUILD_DIR)/release
DEBUG_BIN_DIR := $(DEBUG_DIR)/bin
RELEASE_BIN_DIR := $(RELEASE_DIR)/bin

# Include directories (relative to parent directory)
INC_DIRS := $(SRC_DIR)/inc $(SRC_DIR)/occp $(SRC_DIR)/protocol $(SRC_DIR)/infra
INCLUDES := $(addprefix -I, $(INC_DIRS))

# Compiler Configuration
CXX := g++
CXXFLAGS_BASE := -std=c++17 -Wall -Werror
CXXFLAGS_BASE += -pthread -fPIC
CXXFLAGS_BASE += $(INCLUDES)

CXXFLAGS_EXTRA := -Wextra
CXXFLAGS_EXTRA += -Wformat
CXXFLAGS_EXTRA += -Wformat-security 
CXXFLAGS_EXTRA += -fPIE -pie
CXXFLAGS_EXTRA += -fstack-protector-strong
#CXXFLAGS_EXTRA += -D_GLIBCXX_ASSERTIONS

CXXFLAGS := $(CXXFLAGS_BASE)
CXXFLAGS += $(CXXFLAGS_EXTRA)

# Debug and Release specific flags
DEBUG_FLAGS := $(CXXFLAGS) -g -O0 -DOCCP_DEBUG -fsanitize=address -fsanitize=undefined
RELEASE_FLAGS := $(CXXFLAGS) -O3 -DNDEBUG

# Library dependencies
LIBS := -lpthread

# Source files organization (relative to parent directory)
INC_SRCS := $(wildcard $(SRC_DIR)/inc/*.cpp)
INFRA_SRCS := $(wildcard $(SRC_DIR)/infra/*.cpp)
PROTOCOL_SRCS := $(wildcard $(SRC_DIR)/protocol/*.cpp)
OCCP_SRCS := $(wildcard $(SRC_DIR)/occp/*.cpp)
MAIN_SRC := main.cpp

# All source files
ALL_SRCS := $(INC_SRCS) $(INFRA_SRCS) $(PROTOCOL_SRCS) $(OCCP_SRCS) $(MAIN_SRC)

# Object files for debug and release
# For source files in parent directory, create objects in subdirectories
DEBUG_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/%.o,$(INC_SRCS) $(INFRA_SRCS) $(PROTOCOL_SRCS) $(OCCP_SRCS)) $(DEBUG_DIR)/$(MAIN_SRC:.cpp=.o)
RELEASE_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(RELEASE_DIR)/%.o,$(INC_SRCS) $(INFRA_SRCS) $(PROTOCOL_SRCS) $(OCCP_SRCS)) $(RELEASE_DIR)/$(MAIN_SRC:.cpp=.o)

# Target executables
DEBUG_TARGET := $(DEBUG_BIN_DIR)/$(PROJECT_NAME)
RELEASE_TARGET := $(RELEASE_BIN_DIR)/$(PROJECT_NAME)

# Dependency files
DEBUG_DEPS := $(DEBUG_OBJS:.o=.d)
RELEASE_DEPS := $(RELEASE_OBJS:.o=.d)

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all debug release clean clean-debug clean-release

# Default target
all: debug

# Debug build
debug: $(DEBUG_TARGET)

# Release build
release: $(RELEASE_TARGET)

# Debug executable
$(DEBUG_TARGET): $(DEBUG_OBJS)
	@echo "Linking debug executable: $@"
	@mkdir -p $(DEBUG_BIN_DIR)
	@$(CXX) $(DEBUG_OBJS) -o $@ $(DEBUG_FLAGS) $(LIBS)
	@echo "Debug build complete: $@"

# Release executable
$(RELEASE_TARGET): $(RELEASE_OBJS)
	@echo "Linking release executable: $@"
	@mkdir -p $(RELEASE_BIN_DIR)
	@$(CXX) $(RELEASE_OBJS) -o $@ $(RELEASE_FLAGS) $(LIBS)
	@echo "Release build complete: $@"

# Debug object files from parent directory
$(DEBUG_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling (debug): $<"
	@mkdir -p $(dir $@)
	@$(CXX) $(DEBUG_FLAGS) -MMD -MP -c $< -o $@

# Release object files from parent directory
$(RELEASE_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling (release): $<"
	@mkdir -p $(dir $@)
	@$(CXX) $(RELEASE_FLAGS) -MMD -MP -c $< -o $@

# Debug object files from current directory (main.cpp)
$(DEBUG_DIR)/%.o: %.cpp
	@echo "Compiling (debug): $<"
	@mkdir -p $(dir $@)
	@$(CXX) $(DEBUG_FLAGS) -MMD -MP -c $< -o $@

# Release object files from current directory (main.cpp)
$(RELEASE_DIR)/%.o: %.cpp
	@echo "Compiling (release): $<"
	@mkdir -p $(dir $@)
	@$(CXX) $(RELEASE_FLAGS) -MMD -MP -c $< -o $@

# Include dependency files
-include $(DEBUG_DEPS)
-include $(RELEASE_DEPS)

# Clean targets
clean: clean-debug clean-release

clean-debug:
	@echo "Cleaning debug build artifacts..."
	@rm -rf $(DEBUG_DIR)
	@echo "Debug clean complete."

clean-release:
	@echo "Cleaning release build artifacts..."
	@rm -rf $(RELEASE_DIR)
	@echo "Release clean complete."

# Help target
help:
	@echo "OCCP Project Build System"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build debug version (default)"
	@echo "  debug         - Build debug version with sanitizers"
	@echo "  release       - Build optimized release version"
	@echo "  clean         - Remove all build artifacts"
	@echo "  clean-debug   - Remove debug build artifacts only"
	@echo "  clean-release - Remove release build artifacts only"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Build configurations:"
	@echo "  Debug:   -g -O0 -DDEBUG -fsanitize=address,undefined"
	@echo "  Release: -O3 -DNDEBUG -march=native -flto"
	@echo ""
	@echo "Usage examples:"
	@echo "  make           # Build debug version"
	@echo "  make debug     # Build debug version"
	@echo "  make release   # Build release version"
	@echo "  make clean     # Clean all artifacts"

# Show build information
info:
	@echo "OCCP Project Build Information"
	@echo "=============================="
	@echo "Project:       $(PROJECT_NAME) v$(VERSION)"
	@echo "Compiler:      $(CXX)"
	@echo "Base Flags:    $(CXXFLAGS)"
	@echo "Libraries:     $(LIBS)"
	@echo "Sources:       $(words $(ALL_SRCS)) files"
	@echo "Directories:   $(INC_DIRS)"
	@echo "Debug Target:  $(DEBUG_TARGET)"
	@echo "Release Target: $(RELEASE_TARGET)"
