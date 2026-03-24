//=============================================================================
//
//  CLASS IPOPTSolver
//
//=============================================================================


#ifndef COMISO_IPOPTSOLVER_HH
#define COMISO_IPOPTSOLVER_HH

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_IPOPT_AVAILABLE

//== INCLUDES =================================================================
#include <cstddef>
#include <functional>
#include <vector>
#include <string>

#include <CoMISo/Config/CoMISoDefines.hh>

//== NAMESPACES ===============================================================
namespace COMISO {

//== FORWARDDECLARATIONS ======================================================
class NProblemInterface;
class NConstraintInterface;
struct IPOPTCallbackParameters;

//== CLASS DEFINITION =========================================================

/** \class IPOPTSolver
    Solver for Interior Point optimization problems.

    Solves an interior point problem, given an NProblemInterface
    instance and optionally a set of constraints as well as "lazy
    constraints" via NConstraintInterface.

    Lazy constraints are not active while the initial solution to the
    problem is computed. After the first solution is found, the lazy
    constraints are checked and added to the set of active constraints
    if they are violated. This process is then repeated until all
    constraints are satisfied OR a maximum number of solution attempts
    has been reached. In that case the optimization is started once
    more, with all lazy constraints active.
*/
class COMISODLLEXPORT IPOPTSolver
{
public:

  IPOPTSolver();
  ~IPOPTSolver();

  // *********** OPTIONS **************//

  /*!
    Set options of the underlying ipopt solver.

    For a thorough list and documentation of available options, refer
    to: https://www.coin-or.org/Ipopt/documentation/node40.html
  */
  void set_ipopt_option(std::string, const int&);
  void set_ipopt_option(std::string, const double&);
  void set_ipopt_option(std::string, const std::string&);

  /*!
  Get options of the underlying ipopt solver.

  The type of option {int, double, std::string} needs to be passed as
  template argument.
  */
  template<typename T>
  T get_ipopt_option(std::string option);

  /*!
  Set the maximum number of iterations
  */
  void set_max_iterations(const int _max_iterations);
  int get_max_iterations() const;

  /*!  Set the threshold on the lazy inequality constraint to decide
  if we are near the constraint boundary.
  */
  void set_almost_infeasible_threshold(const double _alm_infsb_thrsh);
  double get_almost_infeasible_threshold() const;

  /*!
  Set the max number of incremental lazy constraint iterations before switching
  to the fully constrained problem.
  \note The default value is 5.
  */
  void set_incremental_lazy_constraint_max_iteration_number
    (const int _incr_lazy_cnstr_max_iter_nmbr);
  int get_incremental_lazy_constraint_max_iteration_number() const;

  /*
  Turn on/off solving the fully constraint problem after exhausting the
  incremental lazy constraint iterations.

  \note The default value of this is true.
  */
  void set_enable_all_lazy_contraints(const bool _enbl_all_lzy_cnstr);
  bool get_enable_all_lazy_contraints() const;

  /*
  Set intermediate callback function object. For the definition of
  IPOPTCallbackParameters include the IPOPTCallbackParameters.hh
  header.

  If the callback function returns false, IPOPT will terminate
  prematurely with the User_Requested_Stop status.
  */
  void set_callback_function
  (std::function<bool(const IPOPTCallbackParameters &)>);

  // ********** SOLVE **************** //

  //! Solve a problem instance with an optional set of constraints.
  //! \throws Outcome
  void solve
  (NProblemInterface* _problem,
   const std::vector<NConstraintInterface*>& _constraints = {});

  //! Same as above with additional lazy constraints that are only
  //! added iteratively to the problem if not satisfied.
  //! \throws Outcome
  void solve
  (NProblemInterface* _problem,
   const std::vector<NConstraintInterface*>& _constraints,
   const std::vector<NConstraintInterface*>& _lazy_constraints);

  //! Get the computed solution energy
  double energy();

private:
  class Impl;
  Impl* impl_;

  // inhibit copy
  IPOPTSolver(const IPOPTSolver&);
  IPOPTSolver& operator=(const IPOPTSolver&);
};

} // namespace COMISO

//=============================================================================
#endif // COMISO_IPOPT_AVAILABLE
//=============================================================================
#endif // COMISO_IPOPTSOLVER_HH defined
//=============================================================================
