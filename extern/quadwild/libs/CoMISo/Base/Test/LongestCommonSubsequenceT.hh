// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_LONGESTCOMMONSUBSEQUENCET_HH_INCLUDED
#define BASE_LONGESTCOMMONSUBSEQUENCET_HH_INCLUDED

#include <algorithm>
#include <vector>
#include <limits>

namespace Test {

/*!
The dynamic programming version of the longest common subsequence algorithm,
O(n*m), n = a.size(), m = b.size() execution and storage complexity.

References:
https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
https://rosettacode.org/wiki/Longest_common_subsequence
*/
template <class VectorT>
class LongestCommonSubsequenceT
{
public:
  typedef VectorT Vector;

  typedef int Index;
  enum { INVALID_INDEX = -1 };

  typedef std::vector<Index> Subsequence;

  struct IndexPair
  {
    Index i, j;

    IndexPair() : i(INVALID_INDEX), j(INVALID_INDEX) {}
    IndexPair(const Index _i, const Index _j) : i(_i), j(_j) {}

    bool i_valid() const { return i != INVALID_INDEX; }
    bool j_valid() const { return j != INVALID_INDEX; }
    bool valid() const { return i_valid() || j_valid(); }
    bool matched() const { return i_valid() && j_valid(); }
    bool boundary() const { return i == 0 || j == 0; }
    bool interior() const { return i > 0 && j > 0; }

    void move_up() { --i; }
    void move_left() { --j; }
    void move_up_left() { move_up(), move_left(); }
  };

  typedef std::vector<IndexPair> IndexPairVector;

public:
  //! Constructor from the two vectors, a and b, to compare
  LongestCommonSubsequenceT(const Vector& _a, const Vector& _b)
    : a_(_a), b_(_b), trc_(_a.size(), _b.size())
  {}

  const Vector& a() const { return a_; }
  const Vector& b() const { return b_; }

  Index a_size() const { return (Index)a_.size(); }
  Index b_size() const { return (Index)b_.size(); }

  /*!
  Trace and record the longest common subsequence (LCS) and return its size.
  The LCS or the difference wrt to a_ and b_ can be extracted correspondingly
  with \ref extract(), \ref removed() and \ref added().
  */
  Index trace()
  {
    auto i = a_size();
    auto j = b_size();
    if (i == 0 || j == 0) // guard against trivial input
      return 0;

    std::vector<typename Trace::Type> stck;
    stck.reserve(i + j);
    do
    {
      if (i == 0 || j == 0 || trace(i, j).done())
      {// boundary case or trace_ij already traced, so pop the stack
        const auto trc_type = stck.back();
        stck.pop_back();
        switch (trc_type)
        {
        case Trace::APPEND :  // popping from an APPEND, go up the tree
          ++i, ++j;
          trace(i, j).set(Trace::APPEND, trace_size(i - 1, j - 1) + 1);
          break;

        case Trace::MOVE_LEFT : // popping from a MOVE_LEFT branch
          ++j, --i;
          stck.push_back(Trace::MOVE_UP); // now push into the MOVE_UP branch
          break;

        case Trace::MOVE_UP : // popping from a MOVE_UP branch, close the node
          {
          ++i;
          const auto size_left = trace_size(i, j - 1);
          const auto size_up = trace_size(i - 1, j);
          // different solutions are possible based on whether we use >= or >
          if (size_left >= size_up)
            trace(i, j).set(Trace::MOVE_LEFT, size_left);
          else
            trace(i, j).set(Trace::MOVE_UP, size_up);
          }
          break;

        default: break;// should not reach this
        }
      }
      else if (a_[i - 1] == b_[j - 1]) // same element?
      {// the result is the LCS(i, j) + a_[i - 1]
        --i, --j;
        stck.push_back(Trace::APPEND);
      }
      else
      {// different elements, so we branch here, going to the left first
        --j;
        stck.push_back(Trace::MOVE_LEFT);
      }
    }
    while (!stck.empty());

    return trace_size(i, j);
  }

  //! Extract the LCS indices wrt to either a_ or b_, or both
  void extract(Subsequence* const _a_sbsqn, Subsequence* const _b_sbsqn) const
  {
    backtrace(_a_sbsqn, _b_sbsqn);
  }

  //! Extract the LCS elements (rather than indices) for convenience.
  void extract(Vector& _sbsqn) const
  {
    Subsequence sbsqn;
    extract(&sbsqn, nullptr);
    _sbsqn.resize(sbsqn.size());
    for (size_t i = 0, n = sbsqn.size(); i < n; ++i)
      _sbsqn[i] = a_[sbsqn[i]];
  }

  //! Extract the difference indices, either from a_ or b_
  void difference(Subsequence* const _a_sbsqn, Subsequence* const _b_sbsqn) const
  {
    Subsequence a_sbsqn, b_sbsqn;
    Subsequence* const a_subsqn_ptr = _a_sbsqn == nullptr ? nullptr : &a_sbsqn;
    Subsequence* const b_subsqn_ptr = _b_sbsqn == nullptr ? nullptr : &b_sbsqn;
    extract(a_subsqn_ptr, b_subsqn_ptr);
    difference(a_subsqn_ptr, b_subsqn_ptr, _a_sbsqn, _b_sbsqn);
  }

  //! Extract the matched/mis-matched indices, from both a_ and b_
  void match(IndexPairVector& _mtch) const
  {
    // extract the LCS indices for both a and b
    Subsequence a_sbsqn, b_sbsqn;
    backtrace(&a_sbsqn, &b_sbsqn);

    // reserve space
    _mtch.reserve(a_size() + b_size() - a_sbsqn.size());

    Index i0 = 0, j0 = 0;

    // extract the mis-matched a_ and b_ indices from (i0, j0) to (_i1, _j1)
    auto mismatch = [&i0, &j0, &_mtch](const Index _i1, const Index _j1)
    {
      // add the mis-matched a_ indices from [i0.._i1)
      for (Index i = i0; i < _i1; ++i)
        _mtch.push_back(IndexPair(i, INVALID_INDEX));

      // add the mis-matched b_ indices from [j0.._j1)
      for (Index j = j0; j < _j1; ++j)
        _mtch.push_back(IndexPair(INVALID_INDEX, j));

      // update i0, j0
      i0 = _i1 + 1;
      j0 = _j1 + 1;
    };

    for (size_t k = 0, l = a_sbsqn.size(); k < l; ++k)
    {
      mismatch(a_sbsqn[k], b_sbsqn[k]); // add mismatched elements
      _mtch.push_back(IndexPair(a_sbsqn[k], b_sbsqn[k])); // add the match
    }

    // add the indices from the last match to the end
    mismatch(a_size(), b_size());
  }

private:
  //! A very simple matrix-storage, enough for what we need here
  template <class ElementT>
  class MatrixT
  {
  public:
    typedef ElementT Element;

  public:
    MatrixT(const size_t _row_nmbr, const size_t _col_nmbr)
      : row_nmbr_(_row_nmbr), col_nmbr_(_col_nmbr)
      , elmns_(row_nmbr_ * col_nmbr_)
    {}

    Element& operator()(const Index _i, const Index _j)
    {
      return elmns_[index(_i, _j)];
    }

    const Element& operator()(const Index _i, const Index _j) const
    {
      return elmns_[index(_i, _j)];
    }

  private:
    size_t row_nmbr_;
    size_t col_nmbr_;
    std::vector<Element> elmns_;

  private:
    size_t index(const Index _row_idx, const Index _col_idx) const
    {
      return _row_idx * col_nmbr_ + _col_idx;
    }
  };

  //! Add a flag to the subsequence to indicate if it has been computed
  class Trace
  {
  public:
    // these types correspond to all algorithm trace options
    enum Type { UNKNOWN, APPEND, MOVE_LEFT, MOVE_UP };

  public:
    Trace() : type_(UNKNOWN) {}

    void set(const Type _type, const Index _size)
    {
      type_ = _type, size_ = _size;
    }

    bool done() const { return type_ != UNKNOWN; }
    Index size() const { return size_; }
    bool append() const { return type_ == APPEND; }

    //! Return the address of the next trace
    IndexPair next(const Index _i, const Index _j) const
    {
      IndexPair ij(_i, _j);
      switch (type_)
      {
      case APPEND :  ij.move_up_left(); break;
      case MOVE_LEFT : ij.move_left(); break;
      case MOVE_UP : ij.move_up(); break;
      default: ;// should not reach this
      }
      return ij;
    }

    IndexPair next(const IndexPair& _ij) const { return next(_ij.i, _ij.j); }

  private:
    Type type_;
    Index size_;
  };

  typedef MatrixT<Trace> TraceMatrix;

private:
  const Vector& a_;
  const Vector& b_;
  TraceMatrix trc_; // O(n*m) subsequence trace storage

private:
  const Trace& trace(const Index _i, const Index _j) const
  {
    return trc_(_i - 1, _j - 1);
  }

  const Trace& trace(const IndexPair& _ij) const { return trace(_ij.i, _ij.j); }

  Trace& trace(const Index _i, const Index _j)
  {
    return trc_(_i - 1, _j - 1);
  }

  Trace& trace(const IndexPair& _ij) { return trace(_ij.i, _ij.j); }

  Index trace_size(const Index _i, const Index _j) const
  {
    if (_i == 0 || _j == 0) // these cases are not traced, size is always 0
      return 0;
    return trace(_i, _j).size();
  }

  Index trace_size(const IndexPair& _ij) const
  {
    return trace_size(_ij.i, _ij.j);
  }

  void backtrace(Subsequence* const _a_sbsqn, Subsequence* const _b_sbsqn) const
  {
    for (IndexPair ij(a_size(), b_size()); ij.interior(); )
    {
      const auto& trc = trace(ij); // the ij-trace
      if (!trc.done())// trace() has not been called/something has gone wrong?
        return; // ... just get out of here, instead of crashing

      ij = trc.next(ij); // this decreases the indices appropriately for append

      if (trc.append()) // append, so add i and/or j
      {
        if (_a_sbsqn != nullptr)
          _a_sbsqn->push_back(ij.i);
        if (_b_sbsqn != nullptr)
          _b_sbsqn->push_back(ij.j);
      }
    }
    if (_a_sbsqn != nullptr)
      std::reverse(_a_sbsqn->begin(), _a_sbsqn->end());
    if (_b_sbsqn != nullptr)
      std::reverse(_b_sbsqn->begin(), _b_sbsqn->end());
  }

  void difference(const Subsequence& _sqnc, const Subsequence& _sbsqn,
    Subsequence& _diff) const
  {
    const auto n = _sqnc.size();
    const auto m = _sbsqn.size();
    _diff.reserve(n - m);
    for (size_t i = 0, j = 0; i < n; ++i)
    {
      if (j < m && i == _sbsqn[j])
        ++j; // i is contained in _sbsqn, so not in diff
      else
        _diff.push_back(i);
    }
  }

  //! Disable copy
  LongestCommonSubsequenceT(const LongestCommonSubsequenceT&);

  //! Disable assignment
  LongestCommonSubsequenceT& operator=(const LongestCommonSubsequenceT&);
};

}// namespace Test

#endif//BASE_LONGESTCOMMONSUBSEQUENCET_HH_INCLUDED
