//=============================================================================
//
//  CLASS IPOPTSolver - IMPLEMENTATION
//
//=============================================================================

//== INCLUDES =================================================================

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_IPOPT_AVAILABLE
//=============================================================================

#include "NProblemInterface.hh"
#include "NConstraintInterface.hh"
#include "BoundConstraint.hh"
#include "CoMISo/Utils/CoMISoError.hh"

#include <Base/Debug/DebConfig.hh>
#include <Base/Debug/DebTime.hh>

#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include "IPOPTProblemInstance.hh"

#include "IPOPTSolver.hh"

//== NAMESPACES ===============================================================

namespace COMISO {

//== IMPLEMENTATION ===========================================================

class IPOPTSolver::Impl
{
public:
  Impl()
    : app_(IpoptApplicationFactory()), // Create an instance of IpoptApplication
      alm_infsb_thrsh_(0.5),
      incr_lazy_cnstr_max_iter_nmbr_(5),
      enbl_all_lzy_cnstr_(true)
  {
    setup_ipopt_defaults();
  }

  void set_ipopt_option(std::string, const int&);
  void set_ipopt_option(std::string, const double&);
  void set_ipopt_option(std::string, const std::string&);

  template <typename T>
  T get_ipopt_option(std::string);

private:
  void setup_ipopt_defaults();

public:
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app_;
  std::function<bool(const IPOPTCallbackParameters &)> intermediate_callback_;

  double alm_infsb_thrsh_;
  int incr_lazy_cnstr_max_iter_nmbr_;
  bool enbl_all_lzy_cnstr_;

private:
  // default ipopt options
  static const std::string ipopt_default_hsl_solver;
  static const int ipopt_default_max_iter;
  static const int ipopt_default_mumps_mem_percent;
};

const std::string IPOPTSolver::Impl::ipopt_default_hsl_solver = "ma57";
const int IPOPTSolver::Impl::ipopt_default_max_iter = 200;
const int IPOPTSolver::Impl::ipopt_default_mumps_mem_percent = 5;

void
IPOPTSolver::Impl::
set_ipopt_option
(std::string option, const int& value)
{
  app_->Options()->SetIntegerValue(option, value);
}

void
IPOPTSolver::Impl::
set_ipopt_option
(std::string option, const double& value)
{
  app_->Options()->SetNumericValue(option, value);
}

void
IPOPTSolver::Impl::
set_ipopt_option
(std::string option, const std::string& value)
{
  app_->Options()->SetStringValue(option, value);
}

template <typename T> T
IPOPTSolver::Impl::
get_ipopt_option(std::string)
{
  // @TODO print warning about unsupported option type!
}

template <> int
IPOPTSolver::Impl::
get_ipopt_option<int>(std::string option)
{
  int value;
  app_->Options()->GetIntegerValue(option, value, "");
  return value;
}

template <> double
IPOPTSolver::Impl::
get_ipopt_option<double>(std::string option)
{
  double value;
  app_->Options()->GetNumericValue(option, value, "");
  return value;
}

template <> std::string
IPOPTSolver::Impl::
get_ipopt_option<std::string>(std::string option)
{
  std::string value;
  app_->Options()->GetStringValue(option, value, "");
  return value;
}

void IPOPTSolver::Impl::setup_ipopt_defaults()
{
  // Switch to HSL if available
#if COMISO_HSL_AVAILABLE
  set_ipopt_option("linear_solver", ipopt_default_hsl_solver);
#else
  set_ipopt_option("linear_solver", "mumps");
#endif

#ifdef DEB_ON
  if (!Debug::Config::query().console())
#endif
  {// Block any output on cout and cerr from Ipopt.
    set_ipopt_option("suppress_all_output", "yes");
  }

#ifdef WIN32
  // Restrict memory to be able to run larger problems on windows
  // with the default mumps solver
  // TODO: find out what this does and whether it makes sense to do it
  set_ipopt_option("mumps_mem_percent",
                   ipopt_default_mumps_mem_percent);
#endif

  // set maximum solver iterations
  set_ipopt_option("max_iter", ipopt_default_max_iter);
}

//-----------------------------------------------------------------------------

IPOPTSolver::IPOPTSolver()
  : impl_(new Impl)
{
}

IPOPTSolver::
~IPOPTSolver()
{
  delete impl_;
}

void
IPOPTSolver::
set_ipopt_option
(std::string option, const int& value)
{
  impl_->set_ipopt_option(option, value);
}

void
IPOPTSolver::
set_ipopt_option
(std::string option, const double& value)
{
    impl_->set_ipopt_option(option, value);
}

void
IPOPTSolver::
set_ipopt_option
(std::string option, const std::string& value)
{
    impl_->set_ipopt_option(option, value);
}

template <typename T> T
IPOPTSolver::
get_ipopt_option(std::string option)
{
  return impl_->get_ipopt_option<T>(option);
}

template int IPOPTSolver::get_ipopt_option<int>(std::string);
template double IPOPTSolver::get_ipopt_option<double>(std::string);
template std::string IPOPTSolver::get_ipopt_option<std::string>(std::string);

void
IPOPTSolver::
set_max_iterations
(const int _max_iterations)
{
  impl_->set_ipopt_option("max_iter", _max_iterations);
}

int
IPOPTSolver::
get_max_iterations() const
{
  return impl_->get_ipopt_option<int>("max_iter");
}

void
IPOPTSolver::
set_almost_infeasible_threshold
(const double _alm_infsb_thrsh)
{
  impl_->alm_infsb_thrsh_ = _alm_infsb_thrsh;
}

double
IPOPTSolver::
get_almost_infeasible_threshold() const
{
  return impl_->alm_infsb_thrsh_;
}

void
IPOPTSolver::
set_incremental_lazy_constraint_max_iteration_number
(const int _incr_lazy_cnstr_max_iter_nmbr)
{
  impl_->incr_lazy_cnstr_max_iter_nmbr_ = _incr_lazy_cnstr_max_iter_nmbr;
}

int
IPOPTSolver::
get_incremental_lazy_constraint_max_iteration_number() const
{
  return impl_->incr_lazy_cnstr_max_iter_nmbr_;
}

void
IPOPTSolver::
set_enable_all_lazy_contraints
(const bool _enbl_all_lzy_cnstr)
{
  impl_->enbl_all_lzy_cnstr_ = _enbl_all_lzy_cnstr;
}

bool
IPOPTSolver::
get_enable_all_lazy_contraints() const
{
  return impl_->enbl_all_lzy_cnstr_;
}

void
IPOPTSolver::
set_callback_function
(std::function<bool(const IPOPTCallbackParameters &)> func)
{
  impl_->intermediate_callback_ = func;
}

static void
throw_ipopt_solve_failure
(Ipopt::ApplicationReturnStatus const status)
{
  DEB_enter_func
  DEB_warning(1, " IPOPT solve failure code is " << status)
  // TODO: we could translate these return codes, but will not do it for now
  //  enum ApplicationReturnStatus
  //    {
  //      Solve_Succeeded=0,
  //      Solved_To_Acceptable_Level=1,
  //      Infeasible_Problem_Detected=2,
  //      Search_Direction_Becomes_Too_Small=3,
  //      Diverging_Iterates=4,
  //      User_Requested_Stop=5,
  //      Feasible_Point_Found=6,
  //
  //      Maximum_Iterations_Exceeded=-1,
  //      Restoration_Failed=-2,
  //      Error_In_Step_Computation=-3,
  //      Maximum_CpuTime_Exceeded=-4,
  //      Not_Enough_Degrees_Of_Freedom=-10,
  //      Invalid_Problem_Definition=-11,
  //      Invalid_Option=-12,
  //      Invalid_Number_Detected=-13,
  //
  //      Unrecoverable_Exception=-100,
  //      NonIpopt_Exception_Thrown=-101,
  //      Insufficient_Memory=-102,
  //      Internal_Error=-199
  //    };
  //------------------------------------------------------
  switch (status)
  {
  case Ipopt::Maximum_Iterations_Exceeded:
    COMISO_THROW(QP_MAXIMUM_ITERATIONS_EXCEEDED);
  case Ipopt::NonIpopt_Exception_Thrown:
    // this could be due to a thrown PROGRESS_ABORTED exception, ...
    PROGRESS_RESUME_ABORT; // ... so check if we need to resume it
  default:
    COMISO_THROW(QP_OPTIMIZATION_FAILED);
  }
}

static void
check_ipopt_status
(Ipopt::ApplicationReturnStatus const _stat)
{
  if (_stat != Ipopt::Solve_Succeeded &&
      _stat != Ipopt::Solved_To_Acceptable_Level)
      throw_ipopt_solve_failure(_stat);
}

void
IPOPTSolver::
solve
(NProblemInterface* _problem,
 const std::vector<NConstraintInterface*>& _constraints)
{
  DEB_time_func_def;
  //----------------------------------------------------------------------------
  // 1. Create an instance of IPOPT NLP
  //----------------------------------------------------------------------------
  Ipopt::SmartPtr<Ipopt::TNLP> np = new IPOPTProblemInstance(_problem, _constraints);
  IPOPTProblemInstance* np2 = dynamic_cast<IPOPTProblemInstance*> (Ipopt::GetRawPtr(np));

  np2->set_callback_function(impl_->intermediate_callback_);

  //----------------------------------------------------------------------------
  // 2. exploit special characteristics of problem
  //----------------------------------------------------------------------------

  DEB_out(2,"exploit detected special properties: ");
  if (np2->hessian_constant())
  {
    DEB_out(2,"*constant hessian* ");
    impl_->app_->Options()->SetStringValue("hessian_constant", "yes");
  }

  if (np2->jac_c_constant())
  {
    DEB_out(2, "*constant jacobian of equality constraints* ");
    impl_->app_->Options()->SetStringValue("jac_c_constant", "yes");
  }

  if (np2->jac_d_constant())
  {
    DEB_out(2, "*constant jacobian of in-equality constraints*");
    impl_->app_->Options()->SetStringValue("jac_d_constant", "yes");
  }
  DEB_out(2,"\n");

  //----------------------------------------------------------------------------
  // 3. solve problem
  //----------------------------------------------------------------------------

  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status = impl_->app_->Initialize();
  COMISO_THROW_if(status != Ipopt::Solve_Succeeded, QP_INITIALIZATION_FAILED);

  status = impl_->app_->OptimizeTNLP( np);

  //----------------------------------------------------------------------------
  // 4. output statistics
  //----------------------------------------------------------------------------
  check_ipopt_status(status);

  // Retrieve some statistics about the solve
  Ipopt::Index iter_count = impl_->app_->Statistics()->IterationCount();
  DEB_out(1,"\n*** IPOPT: The problem solved in "
    << iter_count << " iterations!\n");

  Ipopt::Number final_obj = impl_->app_->Statistics()->FinalObjective();
  DEB_out(1,"\n*** IPOPT: The final value of the objective function is "
    << final_obj << "\n");
}

void
IPOPTSolver::
solve
(NProblemInterface*                        _problem,
 const std::vector<NConstraintInterface*>& _constraints,
 const std::vector<NConstraintInterface*>& _lazy_constraints)
{
  DEB_time_func_def;
  //----------------------------------------------------------------------------
  // 0. Initialize IPOPT Application
  //----------------------------------------------------------------------------

  // Initialize the IpoptApplication and process the options
  auto status = impl_->app_->Initialize();
  COMISO_THROW_if(status != Ipopt::Solve_Succeeded, QP_INITIALIZATION_FAILED);

  bool feasible_point_found = false;
  int  cur_pass = impl_->enbl_all_lzy_cnstr_ ? 1 : 0;
  const int max_passes = impl_->incr_lazy_cnstr_max_iter_nmbr_;

  double acceptable_tolerance = get_ipopt_option<double>("acceptable_tol");

  // copy default constraints
  std::vector<NConstraintInterface*> constraints = _constraints;
  std::vector<bool> lazy_added(_lazy_constraints.size(),false);

  // cache statistics of all iterations
  std::vector<int> n_inf;
  std::vector<int> n_almost_inf;

  while(!feasible_point_found && cur_pass < max_passes)
  {
    ++cur_pass;
    //--------------------------------------------------------------------------
    // 1. Create an instance of current IPOPT NLP
    //--------------------------------------------------------------------------
    Ipopt::SmartPtr<Ipopt::TNLP> np = new IPOPTProblemInstance(_problem, constraints);
    IPOPTProblemInstance* np2 = dynamic_cast<IPOPTProblemInstance*> (Ipopt::GetRawPtr(np));
    // enable caching of solution
    np2->store_solution() = true;

    //--------------------------------------------------------------------------
    // 2. exploit special characteristics of problem
    //--------------------------------------------------------------------------

    DEB_out(2, "detected special properties which will be exploit: ");
    if (np2->hessian_constant())
    {
      DEB_out(2, "*constant hessian* ");
      impl_->app_->Options()->SetStringValue("hessian_constant", "yes");
    }

    if (np2->jac_c_constant())
    {
      DEB_out(2, "*constant jacobian of equality constraints* ");
      impl_->app_->Options()->SetStringValue("jac_c_constant", "yes");
    }

    if (np2->jac_d_constant())
    {
      DEB_out(2, "*constant jacobian of in-equality constraints*");
      impl_->app_->Options()->SetStringValue("jac_d_constant", "yes");
    }
    DEB_out(2, "\n");

    //--------------------------------------------------------------------------
    // 3. solve problem
    //--------------------------------------------------------------------------
    {
      DEB_time_session_def("IPOPT App OptimizeTNLP(np)");
      status = impl_->app_->OptimizeTNLP(np);
    }

    check_ipopt_status(status);

    // check lazy constraints
    n_inf.push_back(0);
    n_almost_inf.push_back(0);
    feasible_point_found = true;
    for (unsigned int i = 0; i < _lazy_constraints.size(); ++i)
    {
      if (lazy_added[i])
        continue;
      NConstraintInterface* lc = _lazy_constraints[i];

      double v = lc->eval_constraint(&(np2->solution()[0]));

      bool inf = false;
      bool almost_inf = false;

      if (lc->constraint_type() == NConstraintInterface::NC_EQUAL)
      {
        v = std::abs(v);
        if (v > acceptable_tolerance)
          inf = true;
        else
          if (v > impl_->alm_infsb_thrsh_)
            almost_inf = true;
      }
      else
        if (lc->constraint_type() == NConstraintInterface::NC_GREATER_EQUAL)
        {
          if (v < -acceptable_tolerance)
            inf = true;
          else
            if (v < impl_->alm_infsb_thrsh_)
              almost_inf = true;
        }
        else
          if (lc->constraint_type() == NConstraintInterface::NC_LESS_EQUAL)
          {
            if (v > acceptable_tolerance)
              inf = true;
            else
              if (v > -impl_->alm_infsb_thrsh_)
                almost_inf = true;
          }

      // infeasible?
      if (inf)
      {
        constraints.push_back(lc);
        lazy_added[i] = true;
        feasible_point_found = false;
        ++n_inf.back();
      }

      // almost violated or violated? -> add to constraints
      if (almost_inf)
      {
        constraints.push_back(lc);
        lazy_added[i] = true;
        ++n_almost_inf.back();
      }
    }
  }

  // no termination after max number of passes?
  if (!feasible_point_found)
  {
    DEB_warning(2, "Could not find a feasible point after " << max_passes - 1
                << " incremental lazy constraint iterations");
    if (!impl_->enbl_all_lzy_cnstr_)
      throw_ipopt_solve_failure(Ipopt::Maximum_Iterations_Exceeded);

    DEB_line(2, "Solving with ALL lazy constraints...");
    ++cur_pass;
    for (unsigned int i = 0; i < _lazy_constraints.size(); ++i)
    {
      if (!lazy_added[i])
        constraints.push_back(_lazy_constraints[i]);
    }
    //--------------------------------------------------------------------------
    // 1. Create an instance of current IPOPT NLP
    //--------------------------------------------------------------------------
    Ipopt::SmartPtr<Ipopt::TNLP> np = new IPOPTProblemInstance(_problem, constraints);
    IPOPTProblemInstance* np2 = dynamic_cast<IPOPTProblemInstance*> (Ipopt::GetRawPtr(np));
    // enable caching of solution
    np2->store_solution() = true;

    //--------------------------------------------------------------------------
    // 2. exploit special characteristics of problem
    //--------------------------------------------------------------------------

    DEB_out(2, "exploit detected special properties: ");
    if (np2->hessian_constant())
    {
      DEB_out(2, "*constant hessian* ");
      impl_->app_->Options()->SetStringValue("hessian_constant", "yes");
    }

    if (np2->jac_c_constant())
    {
      DEB_out(2, "*constant jacobian of equality constraints* ");
      impl_->app_->Options()->SetStringValue("jac_c_constant", "yes");
    }

    if (np2->jac_d_constant())
    {
      DEB_out(2, "*constant jacobian of in-equality constraints*");
      impl_->app_->Options()->SetStringValue("jac_d_constant", "yes");
    }
    std::cerr << std::endl;

    //--------------------------------------------------------------------------
    // 3. solve problem
    //--------------------------------------------------------------------------
    status = impl_->app_->OptimizeTNLP( np);
  }

  //----------------------------------------------------------------------------
  // 4. output statistics
  //----------------------------------------------------------------------------
  check_ipopt_status(status);

  // Retrieve some statistics about the solve
  Ipopt::Index iter_count = impl_->app_->Statistics()->IterationCount();
  DEB_out(1, "\n*** IPOPT: The problem solved in "
    << iter_count << " iterations!\n");

  Ipopt::Number final_obj = impl_->app_->Statistics()->FinalObjective();
  DEB_out(1, "\n*** IPOPT: The final value of the objective function is "
    << final_obj << "\n");

  DEB_out(2, "############# IPOPT with "
          "lazy constraints statistics ###############\n");
  DEB_out(2, "#passes     : " << cur_pass << "( of " << max_passes << ")\n");
  for(unsigned int i=0; i<n_inf.size(); ++i)
    DEB_out(3, "pass " << i << " induced " << n_inf[i]
      << " infeasible and " << n_almost_inf[i] << " almost infeasible\n")
}

double
IPOPTSolver::
energy()
{
  return impl_->app_->Statistics()->FinalObjective();
}

//=============================================================================
} // namespace COMISO
//=============================================================================
#endif // COMISO_IPOPT_AVAILABLE
//=============================================================================
