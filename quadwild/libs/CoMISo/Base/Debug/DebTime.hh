// (C) Copyright 2019 by Autodesk, Inc.

#ifndef BASE_DEBTIME_HH_INCLUDED
#define BASE_DEBTIME_HH_INCLUDED

#include <Base/Utils/StopWatch.hh>
#include <Base/Debug/DebOut.hh>
#include <Base/Utils/Exception.hh>

#include <utility>

#ifdef DEB_ON

namespace Debug {

template <typename DivisorT = int>
class StopWatchSessionT
{
  static constexpr typename std::remove_reference<DivisorT>::type ONE = 1;

public:
  StopWatchSessionT(Enter& _deb, const char* _sssn_name = nullptr,
      const int _deb_lvl = 2, const DivisorT _dvsr = ONE)
      : deb(_deb), sssn_name_(_sssn_name), deb_lvl_(_deb_lvl), dvsr_(_dvsr)
  {
    sw_.start();
  }

  ~StopWatchSessionT() NOEXCEPT(false)
  {
    // TODO: implement "prettier" DEB out if seconds turn into minutes/hours/etc
    DEB_line_if(dvsr_ <= ONE, deb_lvl_,
        sssn_name_ << " took " << sw_.stop() / 1000.0 << " s.");
    DEB_line_if(dvsr_ > ONE, deb_lvl_,
        dvsr_ << " x " << sssn_name_ << " took " << sw_.stop() / 1000.0 / dvsr_
              << " s on average.");
  }

private:
  Enter& deb; // intentional variable name match with the DEB_marcos!
  const char* sssn_name_;
  const int deb_lvl_;
  Base::StopWatch sw_;
  const DivisorT dvsr_;

private:
  // disable copy and assignment
  StopWatchSessionT(const StopWatchSessionT&);
  StopWatchSessionT& operator=(const StopWatchSessionT&);
};

// Factory function used to deduce whether StopWatchSession should store a copy
// or a reference to the divisor.
template <typename DivisorT>
StopWatchSessionT<DivisorT> make_stop_watch_session(
    Enter& _deb, const char* _sssn_name,
    const int _deb_lvl, DivisorT&& _dvsr)
{
  return StopWatchSessionT<DivisorT>(
      _deb, _sssn_name, _deb_lvl, std::forward<DivisorT>(_dvsr));
}

} //namespace Debug

#define DEB_time_session_avg(SSSN, LL, DVSR) PROGRESS_TICK; \
  auto __sw_sssn = make_stop_watch_session(deb, SSSN, LL, DVSR);

#define DEB_time_session(SSSN, LL) PROGRESS_TICK; \
  ::Debug::StopWatchSessionT<int> __sw_sssn(deb, SSSN, LL);

#define DEB_time_session_def(SSSN) PROGRESS_TICK; \
  ::Debug::StopWatchSessionT<int> __sw_sssn(deb, SSSN, 2);

#define DEB_time_func(LL) DEB_enter_func \
  ::Debug::StopWatchSessionT<int> __sw_func(deb, __FUNCTION__, LL);

#define DEB_time_func_def DEB_time_func(2)

#else

#define DEB_time_session_avg(SSSN, LL, DVSR) PROGRESS_TICK;

#define DEB_time_session(SSSN, LL) PROGRESS_TICK;

#define DEB_time_session_def(SSSN) PROGRESS_TICK;

#define DEB_time_func(LL) PROGRESS_TICK;

#define DEB_time_func_def PROGRESS_TICK;

#endif // DEB_ON

#endif//BASE_DEBTIME_HH_INCLUDED
