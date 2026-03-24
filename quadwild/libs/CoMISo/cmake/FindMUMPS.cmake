# - Try to find MUMPS
# Once done this will define
#  MUMPS_FOUND - System has Mumps
#  MUMPS_INCLUDE_DIRS - The Mumps include directories
#  MUMPS_LIBRARY_DIRS - The library directories needed to use Mumps
#  MUMPS_LIBRARIES    - The libraries needed to use Mumps

if (MUMPS_INCLUDE_DIR)
  # in cache already
  SET(MUMPS_FIND_QUIETLY TRUE)
endif (MUMPS_INCLUDE_DIR)

find_path(MUMPS_INCLUDE_DIR 
     NAMES mumps.h dmumps_c.h
     HINTS "$ENV{IPOPT_HOME}/ThirdParty/Mumps/MUMPS/include/"
           "$ENV{IPOPT_HOME}/include/coin/ThirdParty"
           "/usr/include/"
		   "${VS_SEARCH_PATH}Ipopt-3.12.9/include/mumps"
           "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/include/mumps"
   )
   
find_library( MUMPS_LIBRARY_DEBUG
              coinmumpsd dmumpsd coinmumpscd
              HINTS "$ENV{IPOPT_HOME}/lib/"
                    "/usr/lib"
					"${VS_SEARCH_PATH}Ipopt-3.12.9/lib"
                    "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/lib"
                    )
                    
find_library( MUMPS_LIBRARY_RELEASE
              coinmumps dmumps coinmumpsc
              HINTS "$ENV{IPOPT_HOME}/lib/"
                    "/usr/lib"
					"${VS_SEARCH_PATH}Ipopt-3.12.9/lib"
                    "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/lib"
                    )                    

include(SelectLibraryConfigurations)
select_library_configurations( MUMPS )                 
                    
set(MUMPS_INCLUDE_DIRS "${MUMPS_INCLUDE_DIR}" )
set(MUMPS_LIBRARIES "${MUMPS_LIBRARY}" )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MUMPS  DEFAULT_MSG
                                  MUMPS_LIBRARY MUMPS_INCLUDE_DIR)

mark_as_advanced(MUMPS_INCLUDE_DIR MUMPS_LIBRARY )
