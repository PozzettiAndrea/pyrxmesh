include (CoMISoExample)

vci_add_executable (small_symmetric_dirichlet ${sources} ${headers} )

# enable rpath linking
set_target_properties(small_symmetric_dirichlet PROPERTIES INSTALL_RPATH_USE_LINK_PATH 1)

target_link_libraries (small_symmetric_dirichlet
  CoMISo::CoMISo
)
