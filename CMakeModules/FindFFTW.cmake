# - Find FFTW
# Find the native FFTW includes and library
# This module defines
#  FFTW_INCLUDE_DIR, where to find fftw3.h, etc.
#  FFTW_LIBRARIES, the libraries needed to use FFTW.
#  FFTW_FOUND, If false, do not try to use FFTW.
# also defined, but not for general use are
#  FFTW_LIBRARY, where to find the FFTW library.

find_path ( FFTW_INCLUDE_DIR fftw3.h )
find_library ( FFTW_LIBRARY NAMES fftw3 )
find_library ( FFTWF_LIBRARY NAMES fftw3f )

set ( FFTW_LIBRARIES ${FFTW_LIBRARY} ${FFTWF_LIBRARY} )
set ( FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR} )

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args( FFTW DEFAULT_MSG FFTW_LIBRARY FFTW_INCLUDE_DIR )
