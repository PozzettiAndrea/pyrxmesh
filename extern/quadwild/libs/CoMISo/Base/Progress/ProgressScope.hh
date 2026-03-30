// Copyright 2021 Autodesk, Inc. All rights reserved.

#ifndef BASE_PROGRESSSCOPE_HH_INCLUDED
#define BASE_PROGRESSSCOPE_HH_INCLUDED

#include <Base/Progress/ProgressNode.hh>

namespace Progress
{

class Scope
{
public:
  Scope(Node* _node) : node_(_node)
  {
    if (!cntx.phony())
      cntx.enter_node(node_);
  }

  Scope(const Scope&) = delete;

  ~Scope()
  {
    if (!cntx.phony() && cntx.active_node() == node_)
      cntx.exit_node();
  }

  Scope& operator=(const Scope&) = delete;

private:
  Node* node_;
};

} // namespace Progress

#define PROGRESS_SCOPE(NODE) \
  Progress::Scope prgr_scpe(Progress::PROGRESS_NODE_NAME(NODE))

#define PROGRESS_NODE_CLASS_NAME(OPRT) OPRT##_Node
#define PROGRESS_NODE_CLASS_BEGIN(OPRT) \
  namespace Progress \
  { \
  class PROGRESS_NODE_CLASS_NAME(OPRT) : public Node \
  { \
  public: \
    PROGRESS_NODE_CLASS_NAME(OPRT) \
    (const char* const _name, Node* _next = nullptr, Node* _chld = nullptr) \
        : Node(_name, _next, _chld) \
    { \
    } \
    PROGRESS_NODE_CLASS_NAME(OPRT) \
    (const PROGRESS_NODE_CLASS_NAME(OPRT) &) = delete;

#define PROGRESS_NODE_CLASS_DEFINE(OPRT, NAME, ...) \
  PROGRESS_DEFINE_NODE_CUSTOM( \
      Progress::PROGRESS_NODE_CLASS_NAME(OPRT), OPRT, NAME, ##__VA_ARGS__)

#define PROGRESS_NODE_CLASS_END(OPRT, NAME, ...) \
  }; /* end Node descendant class */ \
  } /* end namespace Progress */ \
  PROGRESS_NODE_CLASS_DEFINE(OPRT, NAME, ##__VA_ARGS__)

#define PROGRESS_NODE_CLASS_DATA(OPRT) \
  static_cast<Progress::PROGRESS_NODE_CLASS_NAME(OPRT)&>( \
      *Progress::PROGRESS_NODE_NAME(OPRT))

#define PROGRESS_LOG_DATA(VRBL) _os << #VRBL << LOG_COMMA << VRBL << LOG_COMMA

#endif // BASE_PROGRESSSCOPE_HH_INCLUDED
