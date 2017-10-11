#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdbscan" for configuration ""
set_property(TARGET hdbscan APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hdbscan PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "glib-2.0;m"
  IMPORTED_LOCATION_NOCONFIG "/home/ojmakh/Dropbox/School/phd/code/vocount/lib/lib/libhdbscan.so"
  IMPORTED_SONAME_NOCONFIG "libhdbscan.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdbscan )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdbscan "/home/ojmakh/Dropbox/School/phd/code/vocount/lib/lib/libhdbscan.so" )

# Import target "hdbscan_csample" for configuration ""
set_property(TARGET hdbscan_csample APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hdbscan_csample PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "/home/ojmakh/Dropbox/School/phd/code/vocount/lib/bin/hdbscan_csample"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdbscan_csample )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdbscan_csample "/home/ojmakh/Dropbox/School/phd/code/vocount/lib/bin/hdbscan_csample" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
