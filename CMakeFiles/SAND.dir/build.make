# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/justin/Thesis/topo code/SAND"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/justin/Thesis/topo code/SAND"

# Include any dependencies generated for this target.
include CMakeFiles/SAND.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SAND.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SAND.dir/flags.make

CMakeFiles/SAND.dir/source/SAND.cc.o: CMakeFiles/SAND.dir/flags.make
CMakeFiles/SAND.dir/source/SAND.cc.o: source/SAND.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/justin/Thesis/topo code/SAND/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SAND.dir/source/SAND.cc.o"
	/usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SAND.dir/source/SAND.cc.o -c "/home/justin/Thesis/topo code/SAND/source/SAND.cc"

CMakeFiles/SAND.dir/source/SAND.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SAND.dir/source/SAND.cc.i"
	/usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/justin/Thesis/topo code/SAND/source/SAND.cc" > CMakeFiles/SAND.dir/source/SAND.cc.i

CMakeFiles/SAND.dir/source/SAND.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SAND.dir/source/SAND.cc.s"
	/usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/justin/Thesis/topo code/SAND/source/SAND.cc" -o CMakeFiles/SAND.dir/source/SAND.cc.s

# Object files for target SAND
SAND_OBJECTS = \
"CMakeFiles/SAND.dir/source/SAND.cc.o"

# External object files for target SAND
SAND_EXTERNAL_OBJECTS =

SAND: CMakeFiles/SAND.dir/source/SAND.cc.o
SAND: CMakeFiles/SAND.dir/build.make
SAND: /home/justin/deal.ii-candi/deal.II-v9.2.0/lib/libdeal_II.g.so.9.2.0
SAND: /home/justin/deal.ii-candi/p4est-2.2/DEBUG/lib/libp4est.so
SAND: /home/justin/deal.ii-candi/p4est-2.2/DEBUG/lib/libsc.so
SAND: /usr/lib/x86_64-linux-gnu/libz.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/librol.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtempus.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libmuelu-adapters.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libmuelu-interface.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libmuelu.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/liblocathyra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/liblocaepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/liblocalapack.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libloca.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libnoxepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libnoxlapack.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libnox.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libintrepid2.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libintrepid.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteko.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikos.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosbelos.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosamesos2.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosaztecoo.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosamesos.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosml.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libstratimikosifpack.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libanasazitpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libModeLaplace.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libanasaziepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libanasazi.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libamesos2.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libshylu_nodetacho.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libbelosxpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libbelostpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libbelosepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libbelos.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libml.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libifpack.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libzoltan2.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libpamgen_extras.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libpamgen.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libamesos.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libgaleri-xpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libgaleri-epetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libaztecoo.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libisorropia.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libxpetra-sup.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libxpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libthyratpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libthyraepetraext.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libthyraepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libthyracore.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtrilinosss.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetraext.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetrainout.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libkokkostsqr.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetraclassiclinalg.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetraclassicnodeapi.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtpetraclassic.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libepetraext.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libtriutils.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libshards.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libzoltan.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libepetra.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libsacado.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/librtop.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libkokkoskernels.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchoskokkoscomm.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchoskokkoscompat.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchosremainder.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchosnumerics.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchoscomm.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchosparameterlist.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchosparser.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libteuchoscore.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libkokkosalgorithms.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libkokkoscontainers.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libkokkoscore.so
SAND: /home/justin/deal.ii-candi/trilinos-release-12-18-1/lib/libgtest.so
SAND: /usr/lib/x86_64-linux-gnu/libumfpack.so
SAND: /usr/lib/x86_64-linux-gnu/libcholmod.so
SAND: /usr/lib/x86_64-linux-gnu/libccolamd.so
SAND: /usr/lib/x86_64-linux-gnu/libcolamd.so
SAND: /usr/lib/x86_64-linux-gnu/libcamd.so
SAND: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
SAND: /usr/lib/x86_64-linux-gnu/libamd.so
SAND: /home/justin/deal.ii-candi/hdf5-1.10.5/lib/libhdf5_hl.so
SAND: /home/justin/deal.ii-candi/hdf5-1.10.5/lib/libhdf5.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKBO.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKBool.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKBRep.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKernel.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKFeat.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKFillet.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKG2d.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKG3d.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKGeomAlgo.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKGeomBase.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKHLR.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKIGES.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKMath.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKMesh.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKOffset.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKPrim.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKShHealing.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKSTEP.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKSTEPAttr.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKSTEPBase.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKSTEP209.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKSTL.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKTopAlgo.so
SAND: /home/justin/deal.ii-candi/oce-OCE-0.18.2/lib/libTKXSBase.so
SAND: /usr/local/lib/libblas.a
SAND: /home/justin/deal.ii-candi/slepc-3.13.2/lib/libslepc.so
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libpetsc.so
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libHYPRE.so
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libcmumps.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libdmumps.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libsmumps.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libzmumps.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libmumps_common.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libpord.a
SAND: /home/justin/deal.ii-candi/petsc-3.13.1/lib/libscalapack.a
SAND: /usr/lib/x86_64-linux-gnu/liblapack.so
SAND: /usr/lib/x86_64-linux-gnu/libblas.so
SAND: /home/justin/deal.ii-candi/parmetis-4.0.3/lib/libparmetis.so
SAND: /home/justin/deal.ii-candi/parmetis-4.0.3/lib/libmetis.so
SAND: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
SAND: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
SAND: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
SAND: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
SAND: CMakeFiles/SAND.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/justin/Thesis/topo code/SAND/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SAND"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SAND.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SAND.dir/build: SAND

.PHONY : CMakeFiles/SAND.dir/build

CMakeFiles/SAND.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SAND.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SAND.dir/clean

CMakeFiles/SAND.dir/depend:
	cd "/home/justin/Thesis/topo code/SAND" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/justin/Thesis/topo code/SAND" "/home/justin/Thesis/topo code/SAND" "/home/justin/Thesis/topo code/SAND" "/home/justin/Thesis/topo code/SAND" "/home/justin/Thesis/topo code/SAND/CMakeFiles/SAND.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/SAND.dir/depend

