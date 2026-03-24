// Copyright 2022 Autodesk, Inc. All rights reserved.

#ifndef COMISO_TAGGING_HH
#define COMISO_TAGGING_HH

#include <Base/Debug/DebOut.hh>

#include <limits>
#include <algorithm>

namespace COMISO
{

// A class for storing tags similar to std::vector<bool> but with O(1) operation
// to untag all elements
class Tagging
{
  using Tag = unsigned int;

public:
  Tagging(size_t _tag_nmbr) : tags_(_tag_nmbr, 0), tag_(1) {}

  void set_tagged(size_t _idx) { tags_[_idx] = tag_; }

  bool is_tagged(size_t _idx) const { return tags_[_idx] == tag_; }

  // Reset tags. Typically in O(1), but every  4'294'967'294 th call is in
  // O(size)
  void untag_all()
  {
    if (tag_ < std::numeric_limits<Tag>::max())
      ++tag_;
    else
    {
      for (auto& tag : tags_)
        tag = 0;
      tag_ = 1;
    }
  }

private:
  std::vector<Tag> tags_; // Elements are tagged if tag is larger than base_.
  Tag tag_; // Elements with this tag are considered tagged, elements with
            // smaller tag are considered not tagged. Increasing tag_ allows
            // O(1) resetting of tagged status.

};


} // namespace COMISO

#endif // COMISO_TAGGING_HH

//=============================================================================
