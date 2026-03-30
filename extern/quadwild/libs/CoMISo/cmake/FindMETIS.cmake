# - Try to find METIS
# Once done this will define
#  METIS_FOUND - System has Metis
#  METIS_INCLUDE_DIRS - The Metis include directories
#  METIS_LIBRARY_DIRS - The library directories needed to use Metis
#  METIS_LIBRARIES    - The libraries needed to use Metis

if (METIS_INCLUDE_DIR)
  # in cache already
  SET(METIS_FIND_QUIETLY TRUE)
endif (METIS_INCLUDE_DIR)


find_path(METIS_INCLUDE_DIR NAMES metis.h
     HINTS "$ENV{IPOPT_HOME}/ThirdParty/Metis/metis-4.0/Lib/"
           "$ENV{IPOPT_HOME}/include/coin/ThirdParty/"
           "/usr/include/"
           "/usr/include/metis"
           "/opt/local/include"
           "/opt/local/include/metis"
		   "${VS_SEARCH_PATH}Ipopt-3.12.9/include/metis"
           "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/include/metis"
   )
   
find_library( METIS_LIBRARY_RELEASE
              coinmetis metis
              HINTS "$ENV{IPOPT_HOME}/lib/"
                    "/usr/lib"
                    "/opt/local/lib"
                    "${VS_SEARCH_PATH}Ipopt-3.12.9/lib"
                    "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/lib"
                    )
                    
find_library( METIS_LIBRARY_DEBUG
              coinmetisd metisd
              HINTS "$ENV{IPOPT_HOME}/lib/"
                    "/usr/lib"
                    "/opt/local/lib" 
					"${VS_SEARCH_PATH}Ipopt-3.12.9/lib"
                    "${VS_SEARCH_PATH}Ipopt-3.12.4/Ipopt/MSVisualStudio/v8-ifort/installed/lib"
                    )      
                    
include(SelectLibraryConfigurations)
select_library_configurations( METIS )                   

set(METIS_INCLUDE_DIRS "${METIS_INCLUDE_DIR}" )
set(METIS_LIBRARIES "${METIS_LIBRARY}" )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(METIS  DEFAULT_MSG
                                  METIS_LIBRARY METIS_INCLUDE_DIR)

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY )
