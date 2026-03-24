// (C) Copyright 2019 by Autodesk, Inc.

#ifndef QUALITY_HH_INCLUDED
#define QUALITY_HH_INCLUDED

// This file contains various definitions related to enforcing certain code 
// quality rules, e.g., applying W4

#if defined(_MSC_VER) 

  // Some warnings are disabled permanently since the cannot be controlled on 
  // section basis.

  //function 'function' marked as __forceinline not inlined
  #pragma warning (disable: 4714)

  #define LOW_CODE_QUALITY_SECTION_BEGIN1 __pragma(warning(push, 1))
  #define LOW_CODE_QUALITY_SECTION_BEGIN3 __pragma(warning(push, 3))
  #define LOW_CODE_QUALITY_SECTION_END __pragma(warning(pop))
  #define LOW_CODE_QUALITY_VARIABLE_ALLOW(FOO) static_cast<void>(FOO);

#else // not on MSVC 

  #if __cplusplus >= 201103L

    #ifdef __clang__

      #define LOW_CODE_QUALITY_SECTION_BEGIN1 \
        _Pragma("clang diagnostic push") \
        _Pragma("clang diagnostic ignored \"-Wdeprecated-register\"") \
        _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"") \
        _Pragma("clang diagnostic ignored \"-Wshift-negative-value\"")
        //TODO: ignore more warnings as needed

      #define LOW_CODE_QUALITY_SECTION_END \
        _Pragma("clang diagnostic pop")

    #elif defined(__GNUG__)

      #define LOW_CODE_QUALITY_SECTION_BEGIN1 \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wall\"") //TODO: ignore more warnings as needed

      #define LOW_CODE_QUALITY_SECTION_END \
        _Pragma("GCC diagnostic pop")

    #else

      #define LOW_CODE_QUALITY_SECTION_END

    #endif

  #else // __cplusplus < 201103L

    #define LOW_CODE_QUALITY_SECTION_BEGIN1
    #define LOW_CODE_QUALITY_SECTION_END

  #endif // __cplusplus >= 201103L

  #define LOW_CODE_QUALITY_SECTION_BEGIN3
  #define LOW_CODE_QUALITY_VARIABLE_ALLOW(FOO)  static_cast<void>(sizeof(FOO));

#endif // ifdef _MSC_VER

// backwards-compatible name for LOW_CODE_QUALITY_SECTION_BEGIN1
#define LOW_CODE_QUALITY_SECTION_BEGIN LOW_CODE_QUALITY_SECTION_BEGIN1

#endif // QUALITY_HH_INCLUDED
