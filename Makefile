# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/162/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/162/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/justin/Thesis/topo code/SAND"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/justin/Thesis/topo code/SAND"

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/snap/clion/162/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/snap/clion/162/bin/cmake/linux/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/justin/Thesis/topo code/SAND/CMakeFiles" "/home/justin/Thesis/topo code/SAND//CMakeFiles/progress.marks"
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/justin/Thesis/topo code/SAND/CMakeFiles" 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named info

# Build rule for target.
info: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 info
.PHONY : info

# fast build rule for target.
info/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/info.dir/build.make CMakeFiles/info.dir/build
.PHONY : info/fast

#=============================================================================
# Target rules for targets named strip_comments

# Build rule for target.
strip_comments: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 strip_comments
.PHONY : strip_comments

# fast build rule for target.
strip_comments/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/strip_comments.dir/build.make CMakeFiles/strip_comments.dir/build
.PHONY : strip_comments/fast

#=============================================================================
# Target rules for targets named distclean

# Build rule for target.
distclean: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 distclean
.PHONY : distclean

# fast build rule for target.
distclean/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/distclean.dir/build.make CMakeFiles/distclean.dir/build
.PHONY : distclean/fast

#=============================================================================
# Target rules for targets named runclean

# Build rule for target.
runclean: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 runclean
.PHONY : runclean

# fast build rule for target.
runclean/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runclean.dir/build.make CMakeFiles/runclean.dir/build
.PHONY : runclean/fast

#=============================================================================
# Target rules for targets named SAND

# Build rule for target.
SAND: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 SAND
.PHONY : SAND

# fast build rule for target.
SAND/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/build
.PHONY : SAND/fast

#=============================================================================
# Target rules for targets named debug

# Build rule for target.
debug: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 debug
.PHONY : debug

# fast build rule for target.
debug/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/debug.dir/build.make CMakeFiles/debug.dir/build
.PHONY : debug/fast

#=============================================================================
# Target rules for targets named run

# Build rule for target.
run: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 run
.PHONY : run

# fast build rule for target.
run/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/build
.PHONY : run/fast

#=============================================================================
# Target rules for targets named release

# Build rule for target.
release: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 release
.PHONY : release

# fast build rule for target.
release/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/release.dir/build.make CMakeFiles/release.dir/build
.PHONY : release/fast

source/density_filter.o: source/density_filter.cc.o
.PHONY : source/density_filter.o

# target to build an object file
source/density_filter.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/density_filter.cc.o
.PHONY : source/density_filter.cc.o

source/density_filter.i: source/density_filter.cc.i
.PHONY : source/density_filter.i

# target to preprocess a source file
source/density_filter.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/density_filter.cc.i
.PHONY : source/density_filter.cc.i

source/density_filter.s: source/density_filter.cc.s
.PHONY : source/density_filter.s

# target to generate assembly for a file
source/density_filter.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/density_filter.cc.s
.PHONY : source/density_filter.cc.s

source/kkt_system.o: source/kkt_system.cc.o
.PHONY : source/kkt_system.o

# target to build an object file
source/kkt_system.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/kkt_system.cc.o
.PHONY : source/kkt_system.cc.o

source/kkt_system.i: source/kkt_system.cc.i
.PHONY : source/kkt_system.i

# target to preprocess a source file
source/kkt_system.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/kkt_system.cc.i
.PHONY : source/kkt_system.cc.i

source/kkt_system.s: source/kkt_system.cc.s
.PHONY : source/kkt_system.s

# target to generate assembly for a file
source/kkt_system.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/kkt_system.cc.s
.PHONY : source/kkt_system.cc.s

source/markov_filter.o: source/markov_filter.cc.o
.PHONY : source/markov_filter.o

# target to build an object file
source/markov_filter.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/markov_filter.cc.o
.PHONY : source/markov_filter.cc.o

source/markov_filter.i: source/markov_filter.cc.i
.PHONY : source/markov_filter.i

# target to preprocess a source file
source/markov_filter.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/markov_filter.cc.i
.PHONY : source/markov_filter.cc.i

source/markov_filter.s: source/markov_filter.cc.s
.PHONY : source/markov_filter.s

# target to generate assembly for a file
source/markov_filter.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/markov_filter.cc.s
.PHONY : source/markov_filter.cc.s

source/schur_preconditioner.o: source/schur_preconditioner.cc.o
.PHONY : source/schur_preconditioner.o

# target to build an object file
source/schur_preconditioner.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/schur_preconditioner.cc.o
.PHONY : source/schur_preconditioner.cc.o

source/schur_preconditioner.i: source/schur_preconditioner.cc.i
.PHONY : source/schur_preconditioner.i

# target to preprocess a source file
source/schur_preconditioner.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/schur_preconditioner.cc.i
.PHONY : source/schur_preconditioner.cc.i

source/schur_preconditioner.s: source/schur_preconditioner.cc.s
.PHONY : source/schur_preconditioner.s

# target to generate assembly for a file
source/schur_preconditioner.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/schur_preconditioner.cc.s
.PHONY : source/schur_preconditioner.cc.s

source/watchdog_main.o: source/watchdog_main.cc.o
.PHONY : source/watchdog_main.o

# target to build an object file
source/watchdog_main.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/watchdog_main.cc.o
.PHONY : source/watchdog_main.cc.o

source/watchdog_main.i: source/watchdog_main.cc.i
.PHONY : source/watchdog_main.i

# target to preprocess a source file
source/watchdog_main.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/watchdog_main.cc.i
.PHONY : source/watchdog_main.cc.i

source/watchdog_main.s: source/watchdog_main.cc.s
.PHONY : source/watchdog_main.s

# target to generate assembly for a file
source/watchdog_main.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/SAND.dir/build.make CMakeFiles/SAND.dir/source/watchdog_main.cc.s
.PHONY : source/watchdog_main.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... debug"
	@echo "... distclean"
	@echo "... info"
	@echo "... release"
	@echo "... run"
	@echo "... runclean"
	@echo "... strip_comments"
	@echo "... SAND"
	@echo "... source/density_filter.o"
	@echo "... source/density_filter.i"
	@echo "... source/density_filter.s"
	@echo "... source/kkt_system.o"
	@echo "... source/kkt_system.i"
	@echo "... source/kkt_system.s"
	@echo "... source/markov_filter.o"
	@echo "... source/markov_filter.i"
	@echo "... source/markov_filter.s"
	@echo "... source/schur_preconditioner.o"
	@echo "... source/schur_preconditioner.i"
	@echo "... source/schur_preconditioner.s"
	@echo "... source/watchdog_main.o"
	@echo "... source/watchdog_main.i"
	@echo "... source/watchdog_main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

