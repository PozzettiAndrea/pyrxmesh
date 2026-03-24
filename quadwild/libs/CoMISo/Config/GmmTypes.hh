// (C) Copyright 2014 by Autodesk, Inc.

#ifndef GMMTYPES_HH_INCLUDED
#define GMMTYPES_HH_INCLUDED

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_GMM_AVAILABLE

#include <gmm/gmm_matrix.h>

namespace COMISO_GMM
{  

// Matrix Types
typedef gmm::col_matrix< gmm::wsvector<double> > WSColMatrix;
typedef gmm::row_matrix< gmm::wsvector<double> > WSRowMatrix;
typedef gmm::col_matrix< gmm::rsvector<double> > RSColMatrix;
typedef gmm::row_matrix< gmm::rsvector<double> > RSRowMatrix;
typedef gmm::csc_matrix<double> CSCMatrix;

}//namespace COMISO_GMM

#endif // COMISO_GMM_AVAILABLE

#endif//GMMTYPES_HH_INCLUDED

