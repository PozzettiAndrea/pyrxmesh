# - Try to find SPECTRA
# Once done this will define
#  SPECTRA_FOUND         - System has SPECTRA
#  SPECTRA_INCLUDE_DIRS  - The SPECTRA include directories

if (SPECTRA_INCLUDE_DIR)
  # in cache already
  set(SPECTRA_FOUND TRUE)
  set(SPECTRA_INCLUDE_DIRS "${SPECTRA_INCLUDE_DIR}" )
else (SPECTRA_INCLUDE_DIR)

find_path( SPECTRA_INCLUDE_DIR 
	   NAMES SymEigsSolver.h 
           PATHS $ENV{SPECTRA_DIR}
                 /usr/include/spectra
                 /usr/local/include
                 /usr/local/include/spectra/
                 /opt/local/include/spectra/
                 "${CMAKE_WINDOWS_LIBS_DIR}/general/spectra"
                 "${CMAKE_WINDOWS_LIBS_DIR}/spectra"
                 "${CMAKE_WINDOWS_LIBS_DIR}/spectra/include"
		 "${CMAKE_WINDOWS_LIBS_DIR}/eigen/include"
		  ${PROJECT_SOURCE_DIR}/MacOS/Libs/SPECTRA/include
                  ../../External/include
                  ${module_file_path}/../../../External/include
          )

set(SPECTRA_INCLUDE_DIRS "${SPECTRA_INCLUDE_DIR}" )


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set SPECTRA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(SPECTRA  DEFAULT_MSG
                                  SPECTRA_INCLUDE_DIR)

mark_as_advanced(SPECTRA_INCLUDE_DIR)

endif(SPECTRA_INCLUDE_DIR)
