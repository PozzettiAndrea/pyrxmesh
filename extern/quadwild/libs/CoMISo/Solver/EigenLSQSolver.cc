// Copyright 2021 Autodesk, Inc. All rights reserved.

#include "EigenLSQSolverT_impl.hh"

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#if (COMISO_EIGEN3_AVAILABLE)
//== INCLUDES =================================================================

namespace COMISO
{

template class EigenLSQSolverT<1>;
template class EigenLSQSolverT<2>;
template class EigenLSQSolverT<3>;

} // namespace COMISO

//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
