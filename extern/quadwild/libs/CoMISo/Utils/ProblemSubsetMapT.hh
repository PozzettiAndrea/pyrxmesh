// Copyright 2022 Autodesk, Inc. All rights reserved.

#ifndef COMISO_PROBLEMSUBSETMAPT_HH
#define COMISO_PROBLEMSUBSETMAPT_HH

#include <CoMISo/Solver/SolverBaseT.hh>
#include <CoMISo/Utils/Tagging.hh>
#include <CoMISo/Utils/Tools.hh>

#include <Base/Debug/DebOut.hh>

#include <limits>
#include <algorithm>

namespace COMISO
{

// A class for mapping a sparse set of n indices onto a dense set of indices
// between 0 and n.
// Construction of the object is in O(size) where size is the largest allowed
// index. The object can be reset to be reused for another mapping which is in
// O(1).
template <int DIM>
class ProblemSubsetMapT
{
  using Tag = unsigned int;

  using Point            = typename SolverBaseT<DIM>::Point;
  using LinearEquation   = typename SolverBaseT<DIM>::LinearEquation;
  using LinearTerm       = typename SolverBaseT<DIM>::LinearTerm;
  using Value            = typename SolverBaseT<DIM>::Value;
  using ValueVector      = typename SolverBaseT<DIM>::ValueVector;

  static constexpr auto INVALID_ID = std::numeric_limits<size_t>::max();

public:
  ProblemSubsetMapT(size_t _max_var_nmbr)
      : is_mapped_(_max_var_nmbr), is_fixed_(_max_var_nmbr),
        forward_map_(_max_var_nmbr), backward_map_(), fixed_values_()
  {
  }

  // Reset mapping but keep current fixed values.
  // Typically in O(1), but every  2'147'483'646 th call is in O(size)
  void reset()
  {
    is_mapped_.untag_all();
    backward_map_.clear();
  }

  // Reset mapping as well as current fixed values. Typically in O(n) with n
  // being the number of new fixed values.
  void reset_fixed_values(ValueVector _fixed_values = ValueVector())
  {
    DEB_enter_func;
    reset();
    is_fixed_.untag_all();
    fixed_values_ = std::move(_fixed_values);
    DEB_only(auto size_before = fixed_values_.size());
    // Create list with at most one fixed value for each variable
    sort_unique(fixed_values_, [](const Value& _l, const Value& _r)
        { return _l.var_name < _r.var_name; });
    DEB_only(auto size_after = fixed_values_.size());
    DEB_line_if(size_after != size_before, 3,
        "Removed " << size_before - size_after << " fixed values");
    for (size_t i = 0; i < fixed_values_.size(); ++i)
    {
      set_fixed(fixed_values_[i].var_name);
      forward_map_[fixed_values_[i].var_name] = i;
    }
  }

  // Transform a linear equation into sub problem. Returns the update applied to
  // the const term of _eq, i.e. const_term after = const_term + return value
  Point map(LinearEquation& _eq)
  {
    // Update all terms
    auto const_term_before = _eq.const_term;
    for (auto& term : _eq.linear_terms)
    {
      if (is_fixed(term.var_name))
      {
        // For fixed values update right hand side accordingly
        _eq.const_term -= term.coeff * fixed_point(term.var_name);
        // Invalidate variable for later removal
        term.var_name = INVALID_ID;
      }
      else
        map(term.var_name); // Simply map indices for non-fixed variables
    }

    // remove fixed linear terms
    _eq.linear_terms.erase(
      std::remove_if(_eq.linear_terms.begin(), _eq.linear_terms.end(),
        [](const LinearTerm& _elem) { return _elem.var_name == INVALID_ID; }),
      _eq.linear_terms.end());

    return _eq.const_term - const_term_before;
  }

  // Map vector of variable indices into sub problem
  template <typename T>
  void map(std::vector<T>& _indcs)
  {
    for (auto& idx : _indcs)
      map(idx);
    // remove elements mapped to invalid ids, i.e. fixed variables
    _indcs.erase(
        std::remove(_indcs.begin(), _indcs.end(), static_cast<T>(INVALID_ID)),
        _indcs.end());
  }

  // Map _idx from sub problem into original problem.
  void map_back(size_t& _idx) const
  {
    DEB_error_if(_idx >= backward_map_.size(),
        "Trying to map back idx " << _idx << " but sub problem is of size "
                                  << backward_map_.size());
    DEB_only(size_t idx = _idx);
    _idx = backward_map_[_idx];
    DEB_error_if(!is_mapped(_idx),
        "Variable " << idx << " was mapped back to " << _idx
                    << " which wasn't previously seen during forward mapping");
    DEB_error_if(forward_map_[_idx] != idx,
        "Inconsistency in mapping detected. Sub problem index "
            << idx << " was mapped back to original problem index " << _idx
            << " but problem index " << _idx
            << " was mapped to subproblem index " << forward_map_[_idx]);
  }

  // Returne index mapped back from sub problem to original problem
  size_t mapped_back(size_t _idx) const
  {
    map_back(_idx);
    return _idx;
  }

  // Return the list of fixed values, without duplications
  const ValueVector& fixed_values() const { return fixed_values_; }

private:

  // Returns true iff a mapping for _idx has been generated
  bool is_mapped(size_t _idx) const { return is_mapped_.is_tagged(_idx); }

  // Returns true iff the variable _idx is a fixed point
  bool is_fixed(size_t _idx) const { return is_fixed_.is_tagged(_idx); }

  // Mark _idx as being mapped to a new id
  void set_mapped(size_t _idx) { is_mapped_.set_tagged(_idx); }

  // Mark _index as being a fixed point
  void set_fixed(size_t _idx) { is_fixed_.set_tagged(_idx); }

  const Point& fixed_point(size_t _idx) const
  {
    DEB_error_if(!is_fixed(_idx), "Asking for fix point for index "
                                      << _idx << " which is not a fix point");
    // map _idx to position in fixed values vector
    _idx = forward_map_[_idx];
    return fixed_values_[_idx].point;
  }

  // Map _idx to sub problem. Generate new mapping if it hasn't been mapped yet
  template <typename T>
  void map(T& _idx)
  {
    DEB_error_if(static_cast<size_t>(_idx) > forward_map_.size(),
        "Trying to map index " << _idx << " which is larger than maximum index "
                               << forward_map_.size());
    if (is_fixed(_idx))
      _idx = static_cast<T>(INVALID_ID);
    else
    {
      if (!is_mapped(_idx)) // if it hasn't been mapped yet, create mapping now
      {
        forward_map_[_idx] = backward_map_.size();
        backward_map_.push_back(_idx);
        set_mapped(_idx);
      }
      // Apply mapping to sub problem
      _idx = static_cast<T>(forward_map_[_idx]);
    }
  }
  Tagging is_mapped_;
  Tagging is_fixed_;

  std::vector<size_t> forward_map_; // For mapped variables this maps from
                                    // original index to index in sub problem.
                                    // For fixed variables this maps into the
                                    // vector of fixed values.
  std::vector<size_t> backward_map_; // from sub problem to original

  ValueVector fixed_values_; // Fixed values of original problem
};


} // namespace COMISO

#endif // COMISO_PROBLEMSUBSETMAPT_HH

//=============================================================================
