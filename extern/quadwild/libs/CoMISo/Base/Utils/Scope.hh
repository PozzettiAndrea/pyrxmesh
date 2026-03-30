// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_SCOPE_HH_INCLUDED
#define BASE_SCOPE_HH_INCLUDED

#include <Base/Config/BaseDefines.hh>

#include <functional>
#include <type_traits>
#include <utility>

namespace Scope
{

// Helper class that executes a function when the objects is destroyed.
template <typename FuncT>
class ExitT
{
public:
  explicit ExitT(const FuncT& _func) : func_(_func) {}
  explicit ExitT(FuncT&& _func) : func_(std::move(_func)) {}
  ~ExitT() { func_(); }

private:
  FuncT func_;
};

// Helper function to create an ExitT object that deduces the template type.
// Note we need this because we cannot use decltype on lambdas directly.
template <typename FuncT>
auto make_exit(FuncT&& _func)
{
  return ExitT<std::remove_reference_t<decltype(_func)>>(
      std::forward<FuncT>(_func));
}

// Call FUNC at the end of the current scope.
#define SCOPE_ON_EXIT(FUNC) \
  auto BASE_UNIQUE_NAME(on_exit) = Scope::make_exit(FUNC)


// Create a backup of VAR and restore it at the end of scope.
#define SCOPE_VARIABLE_RESTORE(VAR) \
  SCOPE_ON_EXIT(std::bind([&VAR](const auto& _var) { VAR = _var; }, VAR))


// Set NEW_VAL by using SETTER and restore OLD_VAL at the end of the scope.
#define SCOPE_VARIABLE_CHANGE(SETTER, NEW_VAL, OLD_VAL) \
  SCOPE_ON_EXIT(std::bind(SETTER, OLD_VAL)); \
  SETTER(NEW_VAL)


// Assign NEW_VAL to VAR and restore VAR to its original value at the end of the
// scope.
#define SCOPE_VARIABLE_CHANGE_SIMPLE(VAR, NEW_VAL) \
  SCOPE_VARIABLE_RESTORE(VAR); \
  VAR = NEW_VAL

// Same as above, but works for globally defined variables
#define SCOPE_VARIABLE_CHANGE_GLOBAL(VAR, NEW_VAL) \
  SCOPE_VARIABLE_CHANGE([](const auto& _var) { VAR = _var; }, NEW_VAL, VAR)


} // namespace Scope

//=============================================================================
#endif //BASE_SCOPE_HH_INCLUDED
//=============================================================================

