// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_PROGRESSIREPORT_HH_INCLUDED
#define BASE_PROGRESSIREPORT_HH_INCLUDED

namespace Progress 
{
 
/*!
The default sleep duration (in milliseconds) between subsequent calls of
the Progress::IReport functions.
*/
const int SLEEP_DURATION = 500;

/*!
Interface implemented by the application to report progress updates and answer
abort queries. 

The functions in this interface are called periodically once a computation is
running, and until it is complete. The period between subsequent calls to either
function is >= \ref sleep_duration() and can be controlled with \ref
set_sleep_duration().

\note The interface functions are called in a _separate_ thread, asynchronously 
from the thread where the computations take place. This design prevents the app 
progress updates from blocking the computations.
*/
class IReport
{
public:
  /*!
  Progress update information, supplied to the application function implementing
  \ref IReport::update().
  */
  struct UpdateInfo
  {
    int ppln_stge_nmbr; //!< full pipeline stage number 
    int cmpl_stge_nmbr; //!< completed stage number (<= ppln_stge_nmbr) 
    double stge_frct; //!< current stage completion fraction (in [0..1])
    double ppln_frct; //!< full pipeline completion fraction (in [0..1])
    const char* stge_name; /*!< The current stage name as a non-translated 
      C-string, supplied for debugging purposes. */
  };

public:
  /*!
  Override this function to receive periodically updates on the progress of the 
  current stage and the full pipeline.
  */
  virtual void update(const UpdateInfo& _info) = 0;

  /*!
  Override this function to tell if the current operation needs to be aborted.

  If this function returns true, the computation with which this progress report
  is associated is scheduled to abort. The abort happens shortly afterwards and
  the SDK function executing the computation throws the PROGRESS_ABORTED error
  code. Components and applications should catch this exception and take 
  appropriate clean up actions. 
  */
  virtual bool abort() = 0;
};

/*!
Set the pointer to an object implementing IReport to start the Progress tracking 
thread or nullptr to stop it. 

\note The object pointed by IReport must be kept alive until set_report(nullptr) 
is subsequently called to stop the tracking thread, or crashes might occur. 

\note Make sure set_report(nullptr) is called if a progress IReport object has 
been set before shutting down the main computation thread, or a crash may occur.

Calling the function with a different (or the same) report object pointer, 
without first calling it with nullptr, restarts the tracking.

\note In most cases it is more convenient and safer to use the \ref 
Progress::Session wrapper below instead of calling this function directly.
*/
void set_report(IReport* const _rprt);

/*!
Get the currently set IReport object pointer, or nullptr if not tracking.
*/
IReport* report();

//! An scoped wrapper for Config::set_report().
struct Session
{
  //! Constructor
  Session(IReport& _rprt) { set_report(&_rprt); }
  //! Destructor
  ~Session() { set_report(nullptr); }
};

/*!
Set the progress thread sleep duration (in milliseconds) between subsequent 
calls to IReport functions.  

Increasing this duration makes update() and abort() checks less frequent, but
reduces the CPU time used by the progress tracking thread.

The default value is \ref Default::Progress::SLEEP_DURATION.

\note The set value is clamped from below to 1, i.e., the progress track thread
will always sleep at least for 1ms between subsequent calls to IReport.
*/
void set_sleep_duration(
  const int _slp_drtn_ms //<![in] progress update thread sleep duration in ms
  );

/*!
Get the progress thread sleep duration (in milliseconds) between subsequent 
calls to IReport functions.  
*/
int sleep_duration();

}// namespace Progress

#endif // BASE_PROGRESSIREPORT_HH_INCLUDED
