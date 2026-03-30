// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestChecksumCompletion.hh"
#include <Base/Test/TestChecksum.hh>

#include <sstream>
#include <string>

namespace Test
{
namespace Checksum
{

namespace
{
const char* const SUCCESS = "Success: END";
const char* const FAILURE = "Failure: ";
} // namespace

Completion::Completion() : Object("Completion", L_STABLE) {}

void Completion::record_success()
{
  add(Result::OK, SUCCESS, false);
}

void Completion::record_failure(const std::string& _msg)
{
  std::stringstream mess;
  mess << FAILURE << _msg;
  add(Result::FAILURE, mess.str(), false);
}

bool Completion::success(const std::string& _line)
{
  return _line.find(Checksum::completion.name()) != std::string::npos &&
         _line.find(SUCCESS) != std::string::npos;
}

bool Completion::failure(const std::string& _line)
{
  return _line.find(Checksum::completion.name()) != std::string::npos &&
         _line.find(FAILURE) != std::string::npos;
}

// Register the checksum to check test completion.
Completion completion;

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
