include (CoMISoExample)

vci_add_executable (small_finite_element ${sources} ${headers} )

# enable rpath linking
set_target_properties(small_finite_element PROPERTIES INSTALL_RPATH_USE_LINK_PATH 1)

target_link_libraries (small_finite_element
  CoMISo::CoMISo
)
