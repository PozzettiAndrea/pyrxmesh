// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_SAFESHAREDPTRT_HH_INCLUDED
#define BASE_SAFESHAREDPTRT_HH_INCLUDED

#include <Base/Config/BaseDefines.hh>
#include <Base/Utils/BaseError.hh>

namespace Base
{

/*!
Provide safe data sharing functionality in cases a single object is shared
across multiple copies of Base inside different binaries of the same process.

This class should be instantiated in a specific way:
  1. Set the pointer in one of the instances.
  2. Propagate it to all other instances where it is needed.

\note Using std::shared_ptr<> instead does not work in general as different
binaries in the same process may use different memory heaps. Hence, the shared
data must be destroyed in the same binary where it was created.
*/
template <class T> class SafeSharedPtrT
{
public:
  using Self = SafeSharedPtrT<T>;

  SafeSharedPtrT() {}

  SafeSharedPtrT(const Self& _othr) : ptr_(_othr.ptr_) // leaves own_ == false
  {
  }

  ~SafeSharedPtrT()
  {
    if (own_)
      delete ptr_;
  }

  //! Store the unique instance
  Self& operator=(T* _ptr)
  {
    // TODO: Base error messages cannot be added currently as this introduces
    // binary incompatibility. Add them at the major version upgrade.
    BASE_THROW_ERROR_TODO_if(ptr_ != nullptr, "Data already set");
    BASE_THROW_ERROR_TODO_if(_ptr == nullptr, "Data has not been made");
    ptr_ = _ptr;
    own_ = true;
    return *this;
  }

  //! Set an instance made in another binary
  Self& operator=(const Self& _othr)
  {
    // TODO: Base error messages cannot be added currently as this introduces
    // binary incompatibility. Add them at the major version upgrade.
    BASE_THROW_ERROR_TODO_if(ptr_ != nullptr, "Data already set");
    BASE_THROW_ERROR_TODO_if(_othr.ptr_ == nullptr, "Data has not been made");
    ptr_ = _othr.ptr_;
    return *this;
  }

  T* operator->() { return ptr_; }
  const T* operator->() const { return ptr_; }

  T& operator*() { return *ptr_; }
  const T& operator*() const { return *ptr_; }

private:
  T* ptr_ = nullptr; // shared data
  bool own_ = false; // own data or not
};

} // namespace Base

#endif // BASE_SAFESHAREDPTRT_HH_INCLUDED
