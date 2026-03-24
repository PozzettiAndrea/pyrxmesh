// Copyright 2021 Autodesk, Inc. All rights reserved.

#include "Base/Security/Mandatory.hh"
#include "Base/Code/Quality.hh"

#include "ProgressIReport.hh"

#include <Base/Debug/DebOut.hh>
#include <Base/Utils/BaseError.hh>
#include <Base/Utils/OStringStream.hh>
#include <Base/Utils/Environment.hh>
#include <Base/Test/TestChecksum.hh>
#include <Base/Journal/JournalCppDefs.hh>

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

#ifdef PROGRESS_ON

namespace Progress
{

typedef std::chrono::high_resolution_clock Clock;
typedef Clock::time_point Time;
typedef std::chrono::seconds Seconds;
typedef std::chrono::milliseconds Milliseconds;

} // namespace Progress

#ifdef TEST_ON

namespace Test
{
namespace Checksum
{
using namespace Progress;

struct TrackProgress : public Object
{
  TrackProgress() : Object("Progress::Track::run()", L_PRIME) {}

#ifdef DEB_ON // disable this in Debug & Mixed, cannot be tested reliably
  void record(const TickNumber) {}
#else
  void record(const TickNumber _tick_nmbr)
  {
    // check the check frequency here, since we do not now what is the current
    // update frequency set by the test
    static BASE_THREAD_LOCAL Time last_uptd_time = Clock::now();
    // Setting the frequency affects how often we exercise the progress track
    // test: The higher the frequency, the higher the chance of failing.
    const int FREQUENCY_SECONDS = 10;
    const auto uptd_time = Clock::now();
    const auto updt_drtn_s =
        std::chrono::duration_cast<Seconds>(uptd_time - last_uptd_time).count();
    if (updt_drtn_s < FREQUENCY_SECONDS) // update frequency too high?
      return;                            // ignore it
    last_uptd_time = uptd_time;

    static BASE_THREAD_LOCAL TickNumber last_tick_nmbr = 0;
    if (last_tick_nmbr > 0 && last_tick_nmbr == _tick_nmbr)
      add(Result::ERROR, "Failed Progress Tick Update");
    last_tick_nmbr = _tick_nmbr;
  }
#endif // DEB_ON
};

TrackProgress trck_prgrs;
} // namespace Checksum
} // namespace Test

#endif // TEST_ON

namespace Progress 
{

#ifdef JOURNAL_ON

namespace
{
bool journal_report()
{ // verify if we need to journal report updates
  // NOTE: Turning on this is very risky currently and has a good chance of
  // breaking normal journals, since the output from the Track thread is not
  // synchronized.
  const auto vrbl = System::Environment::variable("JOURNAL_PROGRESS_REPORT");
  return vrbl == "YES";
}
} // namespace

static const bool jrnl_rprt_on = journal_report();

#endif // JOURNAL_ON

struct SleepWake // A sleep with wake functionality
{
  template <class DurationT>
  bool sleep_for(const DurationT& _drtn) 
  {
    std::unique_lock<std::mutex> lock(mtx_);
    end_ = false;
    return !cndt_vrbl_.wait_for(lock, _drtn, [&] { return end_; });
  }

  void wake() 
  {
    std::unique_lock<std::mutex> lock(mtx_);
    end_ = true;
    cndt_vrbl_.notify_all();
  }

private:
  std::condition_variable cndt_vrbl_;
  std::mutex mtx_;
  bool end_ = false;
};

class Track
{
public:
  static Track& modify()
  {
    static BASE_THREAD_LOCAL Track trck(&cntx);
    return trck;
  }

  void set_report(IReport* const _rprt);
  IReport* report() const { return rprt_; }

  void set_sleep_duration(const int _slp_drtn_ms) 
  { 
    slp_drtn_ms_ = std::max(1, _slp_drtn_ms); 
  }

  int sleep_duration() const { return slp_drtn_ms_; }

private:
  Context* cntx_;
  IReport* rprt_ = nullptr;
  int slp_drtn_ms_ = SLEEP_DURATION; 

  enum RunStateType 
  { 
    RST_NONE, 
    RST_WORKING,
    RST_SLEEPING,
    RST_END_REQUEST
  };
  typedef std::atomic<RunStateType> AtomicRunState;

  std::mutex run_stte_mtx_;
  AtomicRunState run_stte_;
  SleepWake slp_wke_;

private:
  explicit Track(Context* _cntx) : cntx_(_cntx), run_stte_(RST_NONE) {}

  // ~Track(): We cannot call set_report(nullptr) safely here if 
  // exit()/quick_exit() are in progress: Since ~RunStateSession() will not 
  // execute in the "run" thread during exit(), wait_for_end_run() will cause 
  // the process to hang.

  void request_end_run()
  {// done, run() will exit after it's next entry to stop()
    std::lock_guard<std::mutex> lock(run_stte_mtx_);
    while (run_stte_ == RST_WORKING) // wait for run() to finish its work
      std::this_thread::yield();
    if (run_stte_ == RST_SLEEPING) // run() is now sleeping ...,
    {
      run_stte_ = RST_END_REQUEST; // ... so issue the stop request
      slp_wke_.wake(); // wake the thread
    }
  }

  void wait_for_end_run()
  {
    while (run_stte_ != RST_NONE) // wait for run() to exit
      std::this_thread::yield();
  }

  bool end_run()
  {
    std::lock_guard<std::mutex> lock(run_stte_mtx_);
    // NOTE: The order of the end condition checks below is important. 
    // In particular, checking for RST_END_REQUEST *must* be first, 
    // otherwise the rprt_ object might be already deleted.
    if (run_stte_ == RST_END_REQUEST || cntx_->phony())
      ;// end run requested by the client or the pipeline is complete
    else if (rprt_->abort()) // the app requests abort() ?
    {
#if JOURNAL_ON // Add some journal output to record progress updates
      if (jrnl_rprt_on && ::Journal::on())
      {
        const auto* actv_node = cntx_->active_node();
        Base::OStringStream strm; 
        strm << "// *** Progress ABORT requested! *** Stage: " << 
          actv_node->name << " tick #: " << actv_node->tick_number();
        ::Journal::stream().os() << strm.str;
        ::Journal::stream().end_line();
      }
#endif//JOURNAL_ON 
      cntx_->request_abort(); // propagate the abort request to the context 
    }
    else // continue running and enter the "working" state
    {
      run_stte_ = RST_WORKING;
      return false;
    }
    return true; // end the run
  }

  void run();
};

void accumulate_ticks(const Node* const _node, const Node* const _actv_node,
  TickNumber& _tick_nmbr, int& _done_node_nmbr, int& _all_node_nmbr, 
  int& _done_wght, int& _all_wght, const bool _done)
{
  if (_node == nullptr)
    return;

  _all_wght += _node->weight(); 
  ++_all_node_nmbr;

  // accumulate tick numbers for done and active nodes only 
  if (_done || _node == _actv_node) 
    _tick_nmbr += _node->tick_number(); 

  if (_done) 
  {
    _done_wght += _node->weight(); // accumulate weight for the done nodes
    ++_done_node_nmbr; // accumulate done node number
  }

  int chld_indx = 0;
  for (auto chld = _node->child(); chld != nullptr; chld = chld->next())
  {
    const auto chld_done = _done || chld_indx < _node->done_child_number();
    accumulate_ticks(chld, _actv_node, _tick_nmbr, 
      _done_node_nmbr, _all_node_nmbr, _done_wght, _all_wght, chld_done);
    ++chld_indx;
  }
}

void Track::run()
{
  struct RunStateSession
  {// make sure the run state is properly cleaned up on exceptions 
    std::mutex& run_stte_mtx_;
    AtomicRunState& run_stte_;
    
    RunStateSession(std::mutex& _run_stte_mtx, AtomicRunState& _run_stte) 
      : run_stte_mtx_(_run_stte_mtx), run_stte_(_run_stte)
    {}

    ~RunStateSession() // clear the state at the end the run 
    { 
      std::lock_guard<std::mutex> lock(run_stte_mtx_);
      run_stte_ = RST_NONE; 
    } 
  };
  
  RunStateSession run_stte_sssn(run_stte_mtx_, run_stte_); // run state session
  
  const auto tick_fraction = [](const TickNumber _tick_nmbr,
                                const TickNumber _tick_nmbr_max)
  {
    if (_tick_nmbr == 0 || _tick_nmbr_max == 0)
      return 0.;
    
    // a "dirty" 2x max ticks growth estimate until we get proper estimations
    auto tick_nmbr_max_estm = _tick_nmbr_max;
    while (_tick_nmbr >= tick_nmbr_max_estm)
      tick_nmbr_max_estm *= 2;
    
    return (double) _tick_nmbr / tick_nmbr_max_estm;
  };
  
  const auto update = [this, tick_fraction]()
  {
    TickNumber ppln_tick_nmbr = 0;
    int done_stge_nmbr = 0, ppln_stge_nmbr = -1; // -1 is to discount the root!
    const auto* actv_node = cntx_->active_node();
    int done_wght = 0, all_wght = 0;
    accumulate_ticks(cntx_->root_node(), actv_node, ppln_tick_nmbr,
      done_stge_nmbr, ppln_stge_nmbr, done_wght, all_wght, false);

    TEST(trck_prgrs, record(ppln_tick_nmbr));

    IReport::UpdateInfo updt_info;
    updt_info.ppln_stge_nmbr = ppln_stge_nmbr;
    updt_info.cmpl_stge_nmbr = done_stge_nmbr;
    updt_info.stge_frct = tick_fraction(actv_node->tick_number(),
      actv_node->tick_number_max());
    updt_info.ppln_frct =
      (done_wght + updt_info.stge_frct * actv_node->weight()) / all_wght;
    updt_info.stge_name = actv_node->name;

#if 0 // Help with debugging by adding the ticks to the name
    std::string name_tmp(updt_info.stge_name);
    name_tmp += " " + std::to_string(actv_node->tick_number()) + ", " +
      std::to_string(actv_node->tick_number_max());
    updt_info.stge_name = name_tmp.c_str();
#endif//
#if JOURNAL_ON // Add some journal output to record progress updates
    if (jrnl_rprt_on && ::Journal::on())
    {// stream into a string to minimize the chance of breaking the journal
      Base::OStringStream strm; 
      strm << "// Stage: " << updt_info.stge_name << ", done: " 
        << updt_info.cmpl_stge_nmbr << "/" << updt_info.ppln_stge_nmbr
        << ", stage fraction: " << updt_info.stge_frct 
        << ", pipeline fraction: " << updt_info.ppln_frct;
      ::Journal::stream().os() << strm.str;
      ::Journal::stream().end_line();
    }
#endif//
    
    rprt_->update(updt_info);
  };

  while (!end_run())
  {
    if (cntx_->active())
      update();
    run_stte_ = RST_SLEEPING;
    slp_wke_.sleep_for(Milliseconds(slp_drtn_ms_));
  }
}

void Track::set_report(IReport* const _rprt)
{
  if (rprt_ != nullptr)
  { // stop any existing tracking
    request_end_run(); // guarantees no more references to *rprt_ from run()
    while (!cntx_->phony())  // force exit for all nodes, inc. the root
      cntx_->exit_node();
    // clear any existing abort request to avoid affecting subsequent tracking
    cntx_->end_abort(); 
    // Wait for a run() thread to end, since request_end_run() merely
    // instruct the run thread to end sleeping.
    wait_for_end_run(); 
  }

  rprt_ = _rprt;
  if (_rprt == nullptr)
    return; // tracking already stopped, nothing more to do 

  // set & begin new tracking 
  cntx_->enter_node(root_node);
  std::thread run_thrd([this] { run(); }); // bind run to this and spawn 
  run_thrd.detach();
}

void set_report(IReport* const _rprt)
{
  Track::modify().set_report(_rprt);
}

IReport* report()
{
  return Track::modify().report();
}

void set_sleep_duration(const int _slp_drtn_ms)
{
  Track::modify().set_sleep_duration(_slp_drtn_ms);
}

int sleep_duration()
{
  return Track::modify().sleep_duration();
}

} // namespace Progress 
#endif // PROGRESS_ON
