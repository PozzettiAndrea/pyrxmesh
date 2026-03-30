# - Try to find SUITESPARSE
# Once done this will define
#  
#  SUITESPARSE_FOUND            - system has SUITESPARSE
#  SUITESPARSE_INCLUDE_DIRS     - the SUITESPARSE include directory
#  SUITESPARSE_LIBRARIES        - Link these to use SUITESPARSE
#  SUITESPARSE_SPQR_LIBRARY     - name of spqr library (necessary due to error in debian package)
#  SUITESPARSE_SPQR_LIBRARY_DIR - name of spqr library (necessary due to error in debian package)
#  SUITESPARSE_LIBRARY_DIR      - Library main directory containing suitesparse libs
#  SUITESPARSE_LIBRARY_DIRS     - all Library directories containing suitesparse libs
#  SUITESPARSE_SPQR_VALID       - automatic identification whether or not spqr package is installed correctly

if (SUITESPARSE_INCLUDE_DIRS)
  # Already in cache, be silent
  SET(SUITESPARSE_FIND_QUIETLY TRUE)
endif (SUITESPARSE_INCLUDE_DIRS)

if( WIN32 )
  # Find cholmod part of the suitesparse library collection

  FIND_PATH( CHOLMOD_INCLUDE_DIR cholmod.h
    PATHS "C:\\libs\\win32\\SuiteSparse\\Include"
    "${VS_SEARCH_PATH}"
    PATH_SUFFIXES suitesparse-4.2.1/include/suitesparse
    suitesparse-metis-for-windows-1.2.2-install/include/suitesparse
    )

  # Add cholmod include directory to collection include directories
  if ( CHOLMOD_INCLUDE_DIR )
    list ( APPEND SUITESPARSE_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIR} )
  endif( CHOLMOD_INCLUDE_DIR )


  # find path suitesparse library
  FIND_PATH( SUITESPARSE_LIBRARY_DIRS
    NAMES amd.lib libamd.lib
    PATHS "C:\\libs\\win32\\SuiteSparse\\libs"
    "${VS_SEARCH_PATH}"
    PATH_SUFFIXES suitesparse-4.2.1/lib64
    suitesparse-metis-for-windows-1.2.2-install/lib64			   )

  # if we found the library, add it to the defined libraries
  if ( SUITESPARSE_LIBRARY_DIRS )
    if ( EXISTS "${SUITESPARSE_LIBRARY_DIRS}/libamd.lib" )
      list ( APPEND SUITESPARSE_LIBRARY_DIRS "${SUITESPARSE_LIBRARY_DIRS}/lapack_blas_windows") # because liblapack.lib lies here
      list ( APPEND SUITESPARSE_LIBRARIES optimized;libamd;optimized;libcamd;optimized;libccolamd;optimized;libcholmod;optimized;libcolamd;optimized;metis;optimized;libspqr;optimized;libumfpack;debug;libamdd;debug;libcamdd;debug;libccolamdd;debug;libcholmodd;debug;libspqrd;debug;libumfpackd;debug;libcolamdd;debug;metisd;optimized;liblapack;debug;liblapackd;optimized;suitesparseconfig;debug;suitesparseconfigd )
    else()
      list ( APPEND SUITESPARSE_LIBRARIES optimized;amd;optimized;camd;optimized;ccolamd;optimized;cholmod;optimized;colamd;optimized;metis;optimized;spqr;optimized;umfpack;debug;amdd;debug;camdd;debug;ccolamdd;debug;cholmodd;debug;spqrd;debug;umfpackd;debug;colamdd;debug;metisd;optimized;blas;optimized;libf2c;optimized;lapack;debug;blasd;debug;libf2cd;debug;lapackd )
    endif()

    if(EXISTS  "${CHOLMOD_INCLUDE_DIR}/SuiteSparseQR.hpp")
      SET(SUITESPARSE_SPQR_VALID TRUE CACHE BOOL "SuiteSparseSPQR valid")
    else()
      SET(SUITESPARSE_SPQR_VALID FALSE CACHE BOOL "SuiteSparseSPQR valid")
    endif()

    if(SUITESPARSE_SPQR_VALID)
      FIND_LIBRARY( SUITESPARSE_SPQR_LIBRARY
        NAMES libspqr
        PATHS ${SUITESPARSE_LIBRARY_DIRS} )
      if (SUITESPARSE_SPQR_LIBRARY)
        list ( APPEND SUITESPARSE_LIBRARIES optimized;libspqr;debug;libspqrd)
      else(SUITESPARSE_SPQR_LIBRARY)
        SET(SUITESPARSE_SPQR_VALID FALSE)
      endif (SUITESPARSE_SPQR_LIBRARY)
    endif()


  endif( SUITESPARSE_LIBRARY_DIRS )

else( WIN32 )
  if( APPLE)
    FIND_PATH( CHOLMOD_INCLUDE_DIR cholmod.h
      PATHS  /opt/local/include/ufsparse )
   
    FIND_LIBRARY(SUITESPARSE_LIBRARY
      NAMES libSuiteSparse.dylib
      PATHS /opt/local/lib)
    message("SUITESPARSE_LIBRARY: ${SUITESPARSE_LIBRARY}")

    FIND_PATH( SUITESPARSE_LIBRARY_DIR
      NAMES libSuiteSparse.dylib
      PATHS /opt/local/lib )

    message("SUITESPARSE_LIBRARY_DIR: ${SUITESPARSE_LIBRARY_DIR}")
    list ( APPEND SUITESPARSE_LIBRARY_DIRS ${SUITESPARSE_LIBRARY_DIR} )

    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_LIBRARY})

  else(APPLE)
    FIND_PATH( CHOLMOD_INCLUDE_DIR cholmod.h
      PATHS /usr/local/include
      /usr/include
      /usr/include/suitesparse/
      ${CMAKE_SOURCE_DIR}/MacOS/Libs/cholmod
      PATH_SUFFIXES cholmod/ CHOLMOD/ )


    FIND_PATH( SUITESPARSE_LIBRARY_DIR
      NAMES libcholmod.so
      PATHS /usr/lib
      /usr/lib64
      /usr/local/lib
      /usr/lib/x86_64-linux-gnu )


  endif(APPLE)

  # Add cholmod include directory to collection include directories
  if ( CHOLMOD_INCLUDE_DIR )
    list ( APPEND SUITESPARSE_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIR} )
  endif( CHOLMOD_INCLUDE_DIR )


  # if we found the library, add it to the defined libraries
  if ( SUITESPARSE_LIBRARY_DIR )

    FIND_LIBRARY(SUITESPARSE_AMD_LIBRARY 
                 NAMES amd
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_AMD_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_CAMD_LIBRARY 
                 NAMES camd
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_CAMD_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_CCOLAMD_LIBRARY 
                 NAMES ccolamd
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_CCOLAMD_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_CHOLMOD_LIBRARY 
                 NAMES cholmod
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_CHOLMOD_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_COLAMD_LIBRARY
                 NAMES cholmod 
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_COLAMD_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_CXSPARSE_LIBRARY
                 NAMES cxsparse 
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_CXSPARSE_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_KLU_LIBRARY
                 NAMES klu 
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_KLU_LIBRARY})
    FIND_LIBRARY(SUITESPARSE_UMFPACK_LIBRARY
                 NAMES umfpack 
                 PATHS ${SUITESPARSE_LIBRARY_DIR})
    list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_UMFPACK_LIBRARY})

    #       list ( APPEND SUITESPARSE_LIBRARIES csparse)
    #       list ( APPEND SUITESPARSE_LIBRARIES spqr)

    # Metis and spqr are optional
    FIND_LIBRARY( SUITESPARSE_METIS_LIBRARY
                  NAMES coinmetis metis
                  PATHS "$ENV{IPOPT_HOME}/lib"
                        ${SUITESPARSE_LIBRARY_DIR} )
    if (SUITESPARSE_METIS_LIBRARY)
      list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_METIS_LIBRARY})
    endif(SUITESPARSE_METIS_LIBRARY)

    if(EXISTS  "${CHOLMOD_INCLUDE_DIR}/SuiteSparseQR.hpp")
      SET(SUITESPARSE_SPQR_VALID TRUE CACHE BOOL "SuiteSparseSPQR valid")
    else()
      SET(SUITESPARSE_SPQR_VALID false CACHE BOOL "SuiteSparseSPQR valid")
    endif()

    if(SUITESPARSE_SPQR_VALID)
      FIND_LIBRARY( SUITESPARSE_SPQR_LIBRARY
        NAMES spqr
        PATHS ${SUITESPARSE_LIBRARY_DIR} )
      if (SUITESPARSE_SPQR_LIBRARY)
        list ( APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_SPQR_LIBRARY})
      endif (SUITESPARSE_SPQR_LIBRARY)
    endif()

  endif( SUITESPARSE_LIBRARY_DIR )

endif( WIN32 )

if (SUITESPARSE_LIBRARY_DIR AND SUITESPARSE_LIBRARIES)
  if(WIN32)
    list (APPEND SUITESPARSE_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIR}/../../UFconfig )
  endif(WIN32)
  SET(SUITESPARSE_FOUND TRUE)
else (SUITESPARSE_LIBRARY_DIR AND SUITESPARSE_LIBRARIES)
  SET( SUITESPARSE_FOUND FALSE )
endif (SUITESPARSE_LIBRARY_DIR AND SUITESPARSE_LIBRARIES)

