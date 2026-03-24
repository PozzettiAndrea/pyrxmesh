include (CoMISoExample)

vci_add_executable (small_exact_constraint_satifaction_example ${sources} ${headers} )

# enable rpath linking
set_target_properties(small_exact_constraint_satifaction_example PROPERTIES INSTALL_RPATH_USE_LINK_PATH 1)

target_link_libraries (small_exact_constraint_satifaction_example
  CoMISo::CoMISo
)
