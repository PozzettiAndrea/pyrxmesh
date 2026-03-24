// (C) Copyright 2021 by Autodesk, Inc.

#ifndef DEFINE_ERROR
#error This file should not be included directly, include TestError.hh instead
#endif

#ifdef TEST_ON

#include <Base/Utils/BaseErrorInc.hh>

// Error codes relating to TestChecksumCompare.(cc|hh)
DEFINE_ERROR(TEST_CHECKSUM_COMPARE_IMPL_NULL,
    "Implementation cannot be null. It should be set using "
    "Test::Checksum::Compare::make() or Test::Checksum::Compare::set().")
DEFINE_ERROR(TEST_CHECKSUM_COMPARE_REPORTS_NOT_SET,
    "Report paths have not been set. This should be done using "
    "Test::Checksum::Compare::set_reports().")
DEFINE_ERROR(TEST_CHECKSUM_COMPARE_DIFFERENCES,
    "Checksum comparison failed: differences found!")

// Error codes relating to TestOutcome.(cc|hh)
DEFINE_ERROR(TEST_OUTCOME_UNEXPECTED, "Unexpected outcome.")

// Error codes related to argument parsing
DEFINE_ERROR(TEST_TOO_FEW_ARGS, "Too few arguments have been supplied.")

// Error codes relating to TestChecksum.(cc|hh)
DEFINE_ERROR(TEST_CHECKSUM_FAILED_OPEN_REPORT,
    "Could not open report file.")

#endif // TEST_ON
