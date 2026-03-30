//=============================================================================
//
//  STRUCT IPOPTCallbackParameters
//
//=============================================================================

#ifndef COMISO_IPOPTCALLBACKPARAMETERS_HH
#define COMISO_IPOPTCALLBACKPARAMETERS_HH

#include <CoMISo/Config/config.hh>

#if COMISO_IPOPT_AVAILABLE

#include <IpTNLP.hpp>

//== TYPE DEFINITION CALLBACK PARAMETERS ======================================
namespace COMISO {
struct IPOPTCallbackParameters {
  Ipopt::AlgorithmMode mode;
  Ipopt::Index iter;
  Ipopt::Number obj_value;
  Ipopt::Number inf_pr;
  Ipopt::Number inf_du;
  Ipopt::Number mu;
  Ipopt::Number d_norm;
  Ipopt::Number regularization_size;
  Ipopt::Number alpha_du;
  Ipopt::Number alpha_pr;
  Ipopt::Index ls_trials;
  const Ipopt::IpoptData* ip_data;
  Ipopt::IpoptCalculatedQuantities* ip_cq;
};
}

#endif // COMISO_IPOPT_AVAILABLE

#endif
