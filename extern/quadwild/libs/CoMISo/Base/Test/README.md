# Base/Test Tools Documentation

## Introduction

The tools in Base/Test are designed to make it easy for the user to perform
baseline comparisons in their C++ tests. This means:

* Storing many kinds of data (known as [checksums](#checksums)) during
  the test execution. Data that might be stored are specific to the project
  being tested, but could involve:
  * The returned value of a function call;
  * Values that one expects to remain constant;
  * Values that one expects to remain constant within a tolerance;
  * The number of vertices/faces on a generated ASM body.
* Comparing these data to stored baselines and displaying the differences.

These tools are framework-agnostic, and thus it should be possible to use them
whether you're using your own testing framework or an externally-maintained C++
framework such as <a href="https://github.com/catchorg/Catch2/tree/v2.x"
target="_blank">Catch2</a> or <a href="https://github.com/google/googletest"
target="_blank">googletest</a>. It is perhaps helpful to think of Base/Test as a
'plugin' for C++ test frameworks.

It is recommended to use CTest as the main test driver, as [some of the
code](#analysing-test-results) is able to parse the CTest log to provide extra
functionality, but this is not a requirement.

Some [additional tools](#additional-tools) are also available to help set up a
test system with Base/Test.

Everything described in this documentation exists within the `Test` namespace
and requires `TEST_ON` to be defined.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## Checksums

### What are checksums?

Checksums are pieces of data that represent an object that you would like to
compare against a baseline, and have the following structure:

| <div style="width:150px">Element</div> | Description                                                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Result                                 | A 'result' of type [`Result`](#the-result-class), which has types `OK`, `WARNING`, `ERROR`, `FAILURE`, `CRASH`, `HANG`.            |
| Data                                   | An associated piece of data of type `std::string`. This string could be an error message, a hash of the object, or something else. |

#### Terminology

A few of the checksum-related terms used in this documentation are defined
below.

| <div style="width:200px">Term</div> | Definition                                                                                                                                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *Checksum*                          | A piece of data with the structure described above that represents an object.                                                                                                           |
| *Checksum algorithm*                | The algorithm used by a *checksum class* to calculate a *checksum*.                                                                                                                     |
| *Checksum class*                    | A class that inherits from `Checksum::Object`, potentially with its own member functions for calculating and storing *checksums*. See [below](#how-are-checksums-stored-and-generated). |
| *Checksum object*                   | An instantiation of a checksum class. Many checksum classes are instantiated as global objects, but this is not true of all.                                                            |
| *`Checksum::Object`*                | The base class for checksum classes, i.e. all checksum classes must inherit from this class.                                                                                            |
| *Record a checksum*                 | Calculate and subsequently store a *checksum*.                                                                                                                                          |

### How are checksums stored and generated?

There is a base class called `Checksum::Object` that is the parent of all
checksum classes. It has the following structure:

| <div style="width:400px">Element</div>               | Description                                                                                                                                                                  |
| :--------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class Object`                                       | name must be supplied<br>[level](#checksum-levels) can be supplied (default = `L_ALL`)                                                                                       |
| `  public:`                                          |                                                                                                                                                                              |
| `    name()`                                         | Returns the name of the current checksum class (e.g. 'Outcome', 'Completion'), set in the constructor.                                                                       |
| `    record(result, data)`                           | Generic function to record a checksum. Data can be of any type with a `Base::IOutputStream` representation.                                                                  |
| `    compare(old_checksum, new_checksum)`            | Virtual function that compares two checksums. Returns a [`Difference`](#the-difference-class) object.                                                                        |
| `    allow()`                                        | Is the checksum generated by this class allowed at this [level](#checksum-levels)?                                                                                           |
| `  protected:`                                       |                                                                                                                                                                              |
| `    add(result, string_data)`                       | Function to write a record of a checksum to `REPORT_FILENAME == "report.txt"` in the [current working directory](#test-io). Usually called by `record()`.                    |
| `    compare_data(old_string_data, new_string_data)` | Virtual function that compares two pieces of checksum data. The default implementation performs a string comparison. Returns a [`Difference`](#the-difference-class) object. |

All checksum classes should live in the `Checksum` namespace and inherit from
the `Checksum::Object` class. Checksums can be generated by instantiations of
these classes (i.e. [checksum objects](#terminology)).

### Checksum classes available in Base/Test

Beyond the parent `Checksum::Object` class, the following checksum classes are
available in Base/Test:

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
#### The `Completion` checksum class

This is used at the end of a test to record a checksum stating that the end was
reached successfully and the time taken.

| <div style="width:400px">Element</div> | Description                                                                                                                                                                                                     |
| :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class Completion : Object`            | name = `Completion`<br>[level](#checksum-levels) = `L_STABLE`                                                                                                                                                   |
| `  public:`                            |                                                                                                                                                                                                                 |
| `    record_end()`                     | Records a checksum signifying the end of the test (`END`) and the time taken.                                                                                                                                   |
| `    static end(line)`                 | Checks if a given line contains a completion checksum.                                                                                                                                                          |
| `    static report_failure(test_dir)`  | Search test directory for a test that has failed to record a completion checksum.                                                                                                                               |

[**Instantiations:**](#the-checksum-registry) `Checksum::completion`
([*global*](#global-checksum-objects))

#### The `Condition` checksum class

This checks the outcome of a Boolean expression `EXPR` and records:

1. a description of `EXPR`;
2. the outcome of `EXPR`; and
3. the location (function, line, file) at which this check was invoked (*note:
   this is ignored during [comparisons](#producing-test-comparison-reports)*).

Examples of `EXPR` are: `a == b`, `a < 5`, `(a / 3)  > 0.24`.

It also counts the number of total and failed checks, which can be written as a
separate checksum at the end of the test.

| <div style="width:400px">Element</div>               | Description                                                                                                                   |
| :--------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| `class Condition : Object`                           | name = `Condition`<br>[level](#checksum-levels) = `L_STABLE`                                                                  |
| `  public:`                                          |                                                                                                                               |
| `    record(condition, code_link, outcome)`          | Records a checksum stating a description of the condition, the outcome of the condition check, and the location of the check. |
| `    record_number()`                                | Records a checksum stating the total number of checks performed and the number of checks that failed.                         |
| `  protected:`                                       |                                                                                                                               |
| `    compare_data(old_string_data, new_string_data)` | Performs the usual comparison, but excludes the code link.                                                                    |

[**Instantiations:**](#the-checksum-registry) `Checksum::condition`
([*global*](#global-checksum-objects))

The macro `CHECK_CONDITION(CONDITION)` (defined in TestChecksumCondition.hh) is provided to simplify the use of this class.

#### The `Debug::Event` checksum class

This is designed to be used only when debugging is enabled (`DEB_ON` is
defined).

It can:

* record debug errors and warnings, along with the locations (function, line,
  file) at which they occur;
* record the total number of errors and warnings seen during the test (typically
  done at the end of a test).

| <div style="width:400px">Element</div>               | Description                                                                                                              |
| :--------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| `class Debug::Event : Object`                        | name must be supplied: either `::Debug::ERROR` or `::Debug::WARNING`<br>[level](#checksum-levels) = `L_ALL`  |
| `  public:`                                          |                                                                                                                          |
| `    record(event_string, code_link)`                | Records a checksum stating a description of the event, whether it is an error or warning, and the location of the check. |
| `    record_number()`                                | Records a checksum stating the total number of debug events that have been noted.                                        |

[**Instantiations:**](#the-checksum-registry) `Checksum::Debug::error`
([*global*](#global-checksum-objects)), `Checksum::Debug::warning`
([*global*](#global-checksum-objects))

#### The `File` checksum class

This records the path of the file, the file size and a hash of the file
contents.

| <div style="width:400px">Element</div> | Description                                                                                     |
| :------------------------------------- | :---------------------------------------------------------------------------------------------- |
| `class File : Object`                  | name must be supplied<br>[level](#checksum-levels) = `L_PRIME`                                  |
| `  public:`                            |                                                                                                 |
| `    record(file_path)`                | Records a checksum stating the path of the file, the file size and a hash of the file contents. |

[**Instantiations:**](#the-checksum-registry) None (see below).

TestChecksumFile.hh contains a macro that can be used to instantiate
`Checksum::File` objects. It must be used within the `Checksum` namespace and
has the following syntax:

```cpp
TEST_CHECKSUM_FILE(VRBL, NAME)
```

This will create a file checksum object with variable name `VRBL` and with name
`NAME-file`. As an example (from ReForm):

```cpp
TEST_CHECKSUM_FILE(mesh_save_chksm, "Export::QuadMesh::save()");
```

produces the line

```cpp
File mesh_save_chksm("Export::QuadMesh::save()-file");
```

#### The `IssueNumber` checksum class

This records and compares the number of issues in a test. An individual instance
of the class should be used for each type of issue.

| <div style="width:400px">Element</div> | Description                                                                                                                                                                                                                                                                            |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class IssueNumber : Object`           | name must be supplied<br>[level](#checksum-levels) can be supplied (default = `L_ALL`)                                                                                                                                                                                                 |
| `  protected:`                         |                                                                                                                                                                                                                                                                                        |
| `    record(issue_number, bad_result)` | If `issue_number > 0`, this records a checksum containing `bad_result` (default = [`Result::ERROR`](#the-result-class)) along with `issue_number`;<br><br>If `issue_number == 0`, this records a checksum containing [`Result::OK`](#the-result-class), along with `issue_number`.     |
| `    compare_data(old_data, new_data)` | This extracts the issue number from the data and marks it as [`Difference::IMPROVED`](#the-difference-class) if the new issue number is lower, [`Difference::EQUAL`](#the-difference-class) if it hasn't changed, or [`Difference::REGRESSED`](#the-difference-class) if it is higher. |

[**Instantiations:**](#the-checksum-registry) None, as it is designed to be
instantiated for each particular issue.

#### The `NumberT<ValueT, CompareT>` checksum class

This compares two numbers `a` and `b` of type `ValueT`, which must have a
stream representation for both `std::istringstream` and `Base::IOutputStream`.
`CompareT` is a class implementing the member function `same(a, b)` that
determines if `a` and `b` have the same value:

* `DefaultCompareT` (default): returns `a == b`.
* `NoCompareT`: always returns `true`.
* `DoubleCompare`: `ValueT` must be equal to `double`, and this returns `|a-b|
  <= tolerance` (`tolerance = 1e-12` by default)

| <div style="width:400px">Element</div>     | Description                                                                                                                                                                                                          |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class NumberT<ValueT, CompareT> : Object` | name must be supplied<br>[level](#checksum-levels) can be supplied (default = `L_ALL`)<br>an instance of `CompareT` can be passed in for comparisons, otherwise `CompareT()` is used                                 |
| `  protected:`                             |                                                                                                                                                                                                                      |
| `    compare_data(old_data, new_data)`     | This extracts the old and new values and compares as follows:<br>1. If `old == new`, return `EQUAL`; otherwise<br>2. If `CompareT` comparison returns `true`, return `NEGLIGIBLE`; otherwise<br>3. Return `UNKNOWN`. |

[**Instantiations:**](#the-checksum-registry) None, as it is a templatised class
that should be instantiated by the user.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
#### The `Outcome` checksum class

This records the success of a (function) call along with any associated error
message.

| <div style="width:400px">Element</div>    | Description                                                                                                                          |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `class Outcome : Object`                  | name = `Outcome`<br>[level](#checksum-levels) = `L_STABLE`                                                                           |
| `  public:`                               |                                                                                                                                      |
| `    record(call, ok, error_message)`     | Records a checksum stating the call that was made, whether the outcome was 'ok' (i.e. successful), and the associated error message. |

[**Instantiations:**](#the-checksum-registry) `Checksum::outcome`
([*global*](#global-checksum-objects))

See the section on [Handling Outcomes](#handling-outcomes) for more
information on how to make use of the `Outcome` checksum class.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
### Global checksum objects

Several of the checksum classes mentioned above have global instantiations. This
is because they are so commonly used that it is convenient to provide the
associated checksum objects directly in Base/Test.

**Note:** As checksum objects must have unique names (see [The Checksum
Registry](#the-checksum-registry)), it is not possible, for example, to create
another instantiation of the `Outcome` class beyond the global
`Checksum::outcome` object.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
### Checksum Levels

[Checksum objects](#checksum-terminlogy) (instantiations of checksum classes)
can be categorised into different levels depending on how useful or stable their
algorithms are in different situations. Tests can be run at a certain 'checksum
level' (determined by `Checksum::run_lvl`), which includes only the checksums
generated by checksum objects of that level or lower. The different checksum
levels are:

| Index | Level      | Which checksums are included?                                                                                     |
| :---: | ---------- | :---------------------------------------------------------------------------------------------------------------- |
| 0     | `L_NONE`   | No checksums are included.                                                                                        |
| 1     | `L_STABLE` | Only checksums generated by checksum objects marked as `L_STABLE` are included.                                   |
| 2     | `L_PRIME`  | Only checksums generated by checksum objects marked as `L_PRIME` or `L_STABLE` are included.                      |
| 3     | `L_ALL`    | All checksums (i.e. those generated by checksum objects marked as `L_ALL`, `L_PRIME` or `L_STABLE`) are included. |

There is no restriction on how custom-defined checksum objects are assigned
these labels.

By default, tests are set to level `L_NONE`, though this can be changed by the
encapsulating framework.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## The `Result` class

The `Result` class holds a representation of a result in the test system, e.g.
for checksums, test runs, etc.

It can describe 6 different types of result, each of which is considered to be a
'worse' result than the previous. The `Result::Type` enum is as follows:

| Index | Result Type | Long Description | Short Description |
| :---: | ----------- | ---------------- | ----------------- |
| 0     | `OK`        | "OK"             | " "               |
| 1     | `WARNING`   | "WARNING"        | "W"               |
| 2     | `ERROR`     | "ERROR"          | "E"               |
| 3     | `FAILURE`   | "FAILURE"        | "F"               |
| 4     | `CRASH`     | "CRASH"          | "C"               |
| 5     | `HANG`      | "HANG"           | "H"               |

The short description is used when reading from and writing to a stream (this
class has a general stream representation), and the long description can be
accessed through the `message()` member function.

The class has the following structure:

| <div style="width:150px">Element</div>  | Description                                                                                                                             |
| :-------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `class Result`                          | result type can be specified (default = `OK`)                                                                                           |
| `  public:`                             |                                                                                                                                         |
| `    message()`                         | Returns the long description of the current result type.                                                                                |
| `    type()`                            | Returns the current result type.                                                                                                        |
| `    ok()`                              | Returns if the current result type is equal to `OK`.                                                                                    |
| `    <`                                 | `rslt1 < rslt2` returns `true` if the result type index of `rslt1` is less than the result type index of `rslt2` and `false` otherwise. |
| `    !=`                                | `rslt1 != rslt2` returns `true` if the result type of `rslt1` is not equal to the result type of `rslt2` and `false` otherwise.         |
| `    +=` <a name='result-add'></a>      | If `rslt1` has a higher result type index than `rslt2`, `rslt1 += rslt2` increases the result type of `rslt1` to that of `rslt2`.       |

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## The `Difference` class

The `Difference` class stores a representation of a difference between two
checksums. It consists of a label (indicating the type of difference) along with
an optional text description.

The label can describe 8 types of difference, each of which is considered to be
a more exaggerated difference than the previous. The `Difference::Type` enum is
as follows:

| Index | Difference Type | Type Description |
| :---: | --------------- | ---------------- |
| 0     | `EQUAL`         | "EQUAL"          |
| 1     | `UNKNOWN`       | "UNKNOWN"        |
| 2     | `IMPROVED`      | "IMPROVED"       |
| 3     | `NEGLIGIBLE`    | "NEGLIGIBLE"     |
| 4     | `SUSPICIOUS`    | "SUSPICIOUS"     |
| 5     | `REGRESSED`     | "REGRESSED"      |
| 6     | `WORKED`        | "WORKED"         |
| 7     | `FAILED`        | "FAILED"         |

The class has the following structure:

| <div style="width:250px">Element</div> | Description                                                                                                                                                                   |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class Difference`                     | Difference type can be specified (default = `EQUAL`)<br>Description can be specified (default = `""`)                                                                         |
| `  public:`                            |                                                                                                                                                                               |
| `    type()`                           | Returns the current difference type.                                                                                                                                          |
| `    equal()`                          | Returns if the current difference type is `EQUAL`.                                                                                                                            |
| `    description()`                    | Returns the text description of the current difference.                                                                                                                       |
| `    type_text()`                      | Returns the type description of the current difference.                                                                                                                       |
| `    static type_text(type)`           | Returns the type description of `type`.                                                                                                                                       |
| `    <`                                | `diff1 < diff2` returns `true` if the difference type index of `diff1` is less than the difference type index of `diff2` and `false` otherwise.                               |
| `    ==`                               | `diff1 == diff2` returns `true` if both the difference type and text description are the same and `false` otherwise.                                                          |
| `    +=`                               | If `diff1` has a higher difference type index than `diff2`, `diff1 += diff2` increases the difference type of `diff1` to that of `diff2`. Text descriptions are concatenated. |

The class has a `Base::IOutputStream` representation.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## The Checksum Registry

The Checksum Registry is a simple map from the name of a checksum object to its
instantiation, and has the type `std::map<std::string, Checksum::Object*>`.

Whenever a checksum class is instantiated, a reference to the resulting checksum
object is added to the registry.

The registry is used to do two things:

* it ensures that all checksum objects have a unique names (an error is thrown
  if two checksum objects are added with the same name);

* it provides access to the `compare()` and `compare_data()` member functions of
  all checksum objects. This can be used by a [report
  executable](#analysing-test-results) via the function
  `compare_from_registry()` provided in TestChecksumCompare.hh to analyse and
  compare the test outputs.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)

## Handling outcomes

Base/Test provides several ways of handling outcomes from function calls.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
### The `Outcome` class

**Note:** Not to be confused with the
[`Checksum::Outcome`](#the-outcome-checksum-class) class.

This is a simple class that holds the basic data required for an 'outcome' of a
call:

* whether the call was successful (`ok`);
* the associated error code (`error_code`);
* the associated error message (`error_message`);
* what to do if the outcome was not what was expected (`unexpected_handler`).

The macro `TEST_OUTCOME_ADD_TYPE(TYPE)` has been provided to allow users to
define conversions from their own outcome types to this class. An example of
this is as follows:

```cpp
namespace
{
Test::Outcome::UnexpectedHandler unexpected_handler = []()
{
  /* Throw an error or use some framework-specific way of reporting an error. */
};
}

TEST_OUTCOME_ADD_TYPE(custom_outcome)
{
   /* The instance of custom_outcome is made available as _oc. */

  auto error_code = /* extract error code (either std::string or int) from _oc */;
  bool ok = /* does _oc represent a successful call? */;
  std::string error_message = /* extract error message from _oc */;

  return Test::Outcome(ok, error_code, error_message, unexpected_handler);
}
```

There is a default 'unexpected handler' provided in the `Outcome` class
(`Outcome::default_unexpected_handler`) that can also be used and which simply
throws a `std::runtime_error()`.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
### Outcome macros

A number of macros have been provided to help deal with recording and handling
different outcomes. In general, their use requires a `TEST_OUTCOME_ADD_TYPE()`
definition for each outcome type that will be handled.

| <div style="width:310px">Macro</div>      | Description                                                                                                                                                                                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CHECK_OUTCOME(CALL)`                     | Writes an [`Outcome` checksum](#the-outcome-checksum-class) to record the outcome of `CALL`. Calls the unexpected handler if `CALL` **failed**.                                                                                                                          |
| `EXPECT_FAILED_OUTCOME(ERROR_CODE, CALL)` | Writes an [`Outcome` checksum](#the-outcome-checksum-class) to record the outcome of `CALL`. Calls the unexpected handler if the error code associated with the outcome is **not equal** to `ERROR_CODE`. This is intended to be used for failures that occur by design. |
| `EXPECT_FAILURE(CALL)`                    | Writes an [`Outcome` checksum](#the-outcome-checksum-class) to record the outcome of `CALL`. Calls the unexpected handler if `CALL` **succeeded**. This is intended to be used for failures that occur because of a known (but unresolved) bug.                          |
| `IGNORE_UNSTABLE_OUTCOME(CALL)`           | No checksum is recorded, and a note is written to stdout to state that the outcome of this call is being ignored. Use this if `CALL` produces different outcomes on different platforms/architectures.                                                                   |

## Test Macros

TestChecksum.hh contains some useful macros that can be utilised in tests to
record checksums.

In the table below, `CHKSM_OBJ` ('checksum object') is the global instantiation
of the checksum class that you wish to use (e.g. `outcome`, `completion`, etc.),
and `RCRD_FN` ('record function') is the specific record function to invoke
(e.g. `record(call, result, message)`, `record_end()`, etc.).

| <div style="width:300px">Macro</div> | Description                                                                                                                                                                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TEST(CHKSM_OBJ, RCRD_FN)`           | First checks that `CHKSM_OBJ` is allowed at this [level](#checksum-levels), and then runs `Checksum::CHKSM_OBJ.RCRD_FN`.<br><br>*Example: `TEST(completion, record_end())` checks if the Completion checksum is allowed using `Checksum::completion.allow()`, and, if so, runs `Checksum::completion.record_end()`.* |
| `TEST_if(CNDT, CHKSM_OBJ, RCRD_FN)`  | If `CNDT` is satisfied, run `TEST(CHKSM_OBJ, RCRD_FN)`.                                                                                                                                                                                                                                                              |

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## Test IO

The `Paths` class in TestPaths.(cc|hh) can be used to help set up your project's
test environment and determine the output directory structure.

It stores:

* the (absolute) path to the current test file;
* the (absolute) path to the parent directory of the current test file; and
* the filename of the current test file.

It also determines the directory in which to store the test output, and switches
the **current working directory** to this directory. Once the test ends, the
current working directory is restored to its original value.

The `Paths` class has the following structure:

| <div style="width:250px">Element</div> | Description                                                                                                                                  |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `class Paths`                          | There are two separate constructors for this class—see below.                                                                                |
| `  public:`                            |                                                                                                                                              |
| `    input_path()`                     | Returns the input test path.                                                                                                                 |
| `    input_directory()`                | Returns the input test directory.                                                                                                            |
| `    filename()`                       | Returns the input test filename (input test path - input test directory).                                                                    |
| `    input_filename_path()`            | Construct an input filename path from the input directory and the supplied filename.                                                         |

The class can be instantiated in one of two ways:

1. By supplying the test root directory (`in_root`), the relative path to the
   test file (`in_rel_path`), and, optionally, the output root directory
   (`out_root`) and the relative path to the test output directory
   (`out_rel_path`).

   The (absolute) path to the test file is `in_path = in_root/in_rel_path`.

   The test output directory, `out_dir`, is determined as follows (where `./` is
   the current working directory):

   * If **neither** `out_root` nor `out_rel_path` exists, set `out_dir = ./`;
   * If **only** `out_root` exists, set `out_dir = out_root/`;
   * If **only** `out_rel_path` exists, set `out_dir = ./out_rel_path/`;
   * If **both** `out_root` and `out_rel_path` exist, set `out_dir =
      out_root/out_rel_path/`.

   If `out_dir` does not exist, it is created (including any parent
   directories). The **current working directory** is then switched to
   `out_dir`.

2. By forwarding the command-line arguments (`argc`, `argv`) from the test
   executable and optionally supplying an output root directory (`out_root`). We
   expect the following format for the command-line arguments:

   ```bash
   > test-bin in_root in_rel_path out_rel_path [test_arguments]
   ```

   Having parsed the command-line arguments, we then proceed as in option 1.

   To facilitate debugging a test within an IDE, we also support the format:

   ```bash
   > test-bin in_path [test arguments]
   ```

   In this case the `Paths` object is set up identically, but no test-specific
   output folder is created, and the test output is stored within the current
   working directory.

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## Handling command-line arguments

In test frameworks like Catch2 it is not easy to pass command-line arguments
directly to tests that are running. Base/Test provides functionality within the
`Args` namespace to work around this (see TestArgs.(cc|hh)). It works by
maintaining a vector of arguments globally in the `Args` namespace that can be
accessed and parsed by tests. The available functions are as follows:

| <div style="width:250px">Function</div>   | Description                                                                                                                                                                                                          |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add(arg)`                                | Adds an argument to the vector.                                                                                                                                                                                      |
| `get()`                                   | Returns the vector of arguments.                                                                                                                                                                                     |
| `collect(argc, argv)`[&dagger;](#dagger1) | Collects test-specific arguments from `argv[]` and returns the number of framework-specific arguments.                                                                                                               |
| `collect_from_string(args)`               | Collects test-specific arguments from a space-delimited string. *Note: This is a simple implementation, and will split by space whether or not the space character is enclosed in quotes or is escaped in some way.* |
| `collect_from_variable(variable)`         | Collects test-specific arguments from a space-delimited string stored in an environment variable. Internally, this calls `collect_from_string()`.                                                                    |

<a name='dagger1'></a>
&dagger; The following format is expected for the command-line arguments:

```sh
  > test-executable <framework-args> [-- <test-args>]
```

where `<framework-args>` are the arguments, such as the test name, to be passed
to the test framework (e.g. Catch2, googletest) and `<test-args>` are arguments
to be passed directly to the test being run.

---

The `Test::Args` namespace also provides simple functionality to help parse
these arguments, as follows (see TestArgs.hh for a more detailed description):

### `Args::Option`

The `Args::Option` class is provided as the base class for test command-line
options. Every option that the user intends to parse must have an associated
class derived from `Args::Option` that implements the member function
`do_parse()`. Options must be constructed with:

  * the option's name (e.g. `--option` or `-O`);
  * the minimum number of required arguments (e.g. 2);
  * (optional) a shared pointer to an `Args::Option::Mutex` object (a simple
    wrapper around a mutex flag), if the option exists as part of a mutual
    exclusion group.

**Note:** the option's name should not be `--help` or `-h`, as these names are
reserved by the parser.

`do_parse()` is a virtual function to parse the arguments for the current
option. The return value should be the number of arguments that have been
successfully parsed. The provided macros `OPTION_THROW[_if]()` should be used to
report errors.

### `Args::ToggleGroup`

`Args::ToggleGroup` is a class that wraps mutually-exclusive 'on' and 'off'
options for a particular feature and must be constructed with a `name` and a
`description`.

The names of the created options will be: `--{name}-on` and `--{name}-off`.

The respective descriptions of the created options will be:
* `Enable {description}. Mutually-exclusive with --{name}-off.`
* `Disable {description}. Mutually-exclusive with --{name}-on.`

The options can be added to the parser using the
`Args::Parser::add_option_group()` function.

### Predefined options

A few commonly used options have been implemented in Base/Test and are available
for use. They are listed below with their associated usages/descriptions:

* `Args::JournalToggleGroup`:
  * `--journal-off = Disable the journal. Mutually exclusive with --journal-on.`
  * `--journal-on = Enable the journal. Mutually exclusive with --journal-off.`
* `Args::ProgressToggleGroup`:
  * `--progress-off = Disable progress tracking. Mutually exclusive with --progress-on.`
  * `--progress-on = Enable progress tracking. Mutually exclusive with --progress-off.`
* `Args::DebugLevel` (supply `<dflt>` to constructor):
  * `--debug-level <n> = Set debug level to n (n < 0 disables debug output).
    Default = <dflt>.`
* `Args::ChecksumLevel`:
   * `--checksum-level <L> [force|max] = Adjust the checksum run level specified
     by each test. Use 'force' (default) to set the level to L, and use 'max' to
     reduce any level above L to L, where L = NONE|STABLE|PRIME|ALL.`

### `Args::Parser`

The `Args::Parser` class performs parsing of the arguments in the vector
returned by `Args::get()`. Options may be added to an instance of the parser
using the `add_option()` or `add_option_group()` member functions.

The parser's behaviour can be altered through the use of flags:

| <div style="width:200px">Flag</div> | Description                                                                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FL_NONE`                           | No flags are set.                                                                                                                                                   |
| `FL_OVERWRITE`                      | Allow options to be overwritten (i.e. set more than once). If this flag is not set, an error will be thrown if there is an attempt to set an option more than once. |
| `FL_ALLOW_UNKNOWN` (default)        | Ignore unknown options. If this flag is not set, an error will be thrown if an unknown option is encountered.                                                       |

<details>
  <summary>Why is <code>FL_ALLOW_UNKNOWN</code> set by default?</summary>

---
Different options may be parsed at different points during test execution,
and thus it is sensible to ignore options that are not relevant at the
current point (rather than throwing an error).

Example: In ReForm Preprocess, an `Environment` class sets up the infrastructure
for running the tests. In its constructor it parses the command-line arguments
to set the [checksum level](#checksum-levels) for the tests and the debug output
level (amongst other things).

However, by setting `FL_ALLOW_UNKNOWN`, we allow additional options to be passed
to the test itself.
---
</details>

If the parser encounters `-h` or `--help` when parsing options, it will generate
and display help text for all of the options added to it.

### Example

The following C++ code shows how these classes might be used:

```cpp
// Set some default values
bool journal_on = true;
bool debug_output_on = true;
Checksum::run_lvl = Checksum::Level::L_ALL;

// Create the argument parser
Args::Parser parser();

// Instantiate the relevant options
Args::JournalToggleGroup journal_toggle;
Args::DebugLevel debug_level(2); // Default level = 2
Args::ChecksumLevel checksum_level;

// Add options to the parser
parser.add_option_group(journal_toggle);
parser.add_option(debug_level);
parser.add_option(checksum_level);

// Parse the command-line arguments stored in the Args::get() vector
parser.parse();

// Determine if journaling was switched on/off by a command-line option
if (journal_toggle.parsed())
  journal_on = journal_toggle.on();

// Determine if the debug output level was set by a command-line option
if (debug_level.parsed())
  debug_output_on = debug_level.enabled();

// Set journaling
if (journal_on)
{
  /* Switch journaling on. */
}
else
{
  /* Switch journaling off. */
}

// Set debug output
if (debug_output_on)
{
  int output_level = debug_level.parsed() ? debug_level.level()
                                          : debug_level.default_level();

  /* Switch debug output on and set level to output_level. */
}
else
{
  /* Switch debug output off. */
}

// Modify checksum run level
if (checksum_level.parsed())
{
  Checksum::run_lvl = checksum_level.flag() == Args::ChecksumLevel::FL_FORCE
                          ? checksum_level.level() // FL_FORCE
                          : std::min(checksum_level.level(),
                                default_checksum_level); // FL_MAX
}
```

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
## Analysing Test Results

Once the tests have run and all checksums have been recorded, the results can be
analysed. This is handled by functions in TestReport.(cc|hh) and
TestResultAnalysis.(cc|hh).

### Creating a simple report executable

It is currently intended that a separate 'report' executable be built that
performs analysis of the results. It is straightforward (and recommended) to do
this by creating the following files (or something similar). report.cc will need
to be part of a new executable that links both to the existing project (which we
will henceforth call 'PROJECT') and (statically) to Base. test_compare.(cc|hh)
should be included as part of PROJECT (which will also need to be statically
linked to Base).

**report.cc** (part of report executable)
```cpp
#ifdef TEST_ON

#include <Base/Test/TestReport.hh>
#include <PROJECT/test_compare.hh>

int main(int _argc, const char* _argv[])
{
  return Test::report(_argc, _argv, Test::Checksum::compare);
}

#else // TEST_ON

int main()
{
   return 0;
}

#endif // TEST_ON
```

**PROJECT/test_compare.hh** (part of PROJECT)
```cpp
#pragma once

#ifdef TEST_ON

#include <Base/Test/TestChecksum.hh>
#include <string>

namespace Test
{
namespace Checksum
{

// PROJECT-specific checksum comparison
Difference compare(
    const std::string& _name, const Record& _old_rcrd, const Record& _new_rcrd);

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
```

**PROJECT/test_compare.cc** (part of PROJECT)
```cpp
#include <Base/Test/TestChecksumCompare.hh>
#include "test_compare.hh"

#ifdef TEST_ON

namespace Test
{
namespace Checksum
{

// Wrapper for Base/Test comparison function
Difference compare(
    const std::string& _name, const Record& _old_rcrd, const Record& _new_rcrd)
{
  return compare_from_registry(_name, _old_rcrd, _new_rcrd);
}

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
```

**Note:** On first sight it might appear that the `compare()` function in
test_compare.(cc|hh) is simply a wrapper and thus redundant, as we could simply
use `compare_from_registry()` directly in report.cc instead of `compare()`.
However, this wrapper *is* in fact necessary:

Both the report exectuable and the PROJECT library/executable/package need to be
(statically) linked to a copy of Base. If additional checksum classes are added
to the PROJECT, they will only be accessible through the copy of Base linked to
the PROJECT, *not* the one linked to the report executable. As long as the
test_compare.(cc|hh) files are built as part of the PROJECT, use of the
`compare()` function ensures that the `compare_from_registry()` function used by
the report executable will be the one linked to the PROJECT.

### The `report()` function

Internally, `report()` makes use of the function `make_comparison(root_dir0,
root_dir1, output_type, comparison_func, show_progress)`, which compares the
checksums stored within two directories (usually a test directory and a baseline
directory) (see [Producing test comparison
reports](#producing-test-comparison-reports)). The output it produces depends on
the variable `output_type`, which is of type `CompareOutputType`, an enum
defined as follows:

| Index | Result Type                | Description                                                                                                                                                                         |
| :---: | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | `COT_SHORT_DIFF` (default) | Produce comparison reports that only include checksums that are different between `root_dir0` and `root_dir1`.                                                                      |
| 1     | `COT_FULL_DIFF`            | Produce comparison reports that include all checksums, even those that are identical between `root_dir0` and `root_dir1`.                                                           |
| 2     | `COT_UPDATE`               | Copy all reports from `root_dir1` to `root_dir0` (potentially overwriting reports in `root_dir0`). Any reports that exist in `root_dir0` but not in `root_dir1` are **unaffected**. |
| 3     | `COT_MIRROR`               | Copy all reports from `root_dir1` to `root_dir0` (potentially overwriting reports in `root_dir0`). Any reports that exist in `root_dir0` but not in `root_dir1` are **deleted**.    |

`comparison_func` is a pointer to a function of type `Checksum::Compare` that
can be used to compare two checksums.

Setting `show_progress == false` reduces the amount of progress-related output
from the function.

<details>
  <summary>What is <code>Checksum::Compare</code>?</summary>

---
`Checksum::Compare` is defined in TestChecksumCompare.hh and describes a
function that takes the following 3 arguments and returns a
[`Difference`](#the-difference-class) object:

  * The name of the relevant checksum class;
  * The old checksum; and
  * The new checksum.

One such function is `compare_from_registry()`, defined in
TestChecksumCompare.hh.
---

</details>

`report()` takes 3 arguments:

1. The number of command-line arguments (`argc`);
2. The command-line argument values (`argv`); and
3. A pointer to a function of type `Checksum::Compare` (see above).

It expects and parses the following command-line format for `argv`:

```bash
> <report-executable> dir_left dir_right [--full-diff|--update|--mirror] [--no-progress]
```

It then runs `make_comparison()` on `dir_left` and `dir_right`. By default,

* `output_type = COT_SHORT_DIFF` and
* `show_progress = true`,

though these can be modified through use of the optional arguments.

### Creating a custom report executable

Use of the `report()` function is the recommended approach for creating a report
executable simply because [other helpful tools](#additional-tools) have been
written that expect its command-line format.

However, it is possible to produce a report executable without using the
`report()` function. Such an executable may make use of the `make_comparison()`
function described above (and exposed in TestReport.hh).

[comment]: # (If the heading below is changed, all links to this section must also be updated.)
### Producing test comparison reports

`make_comparison(root_dir0, root_dir1, output_type, comparison_func)` can be
used to compare two test directories: the left suite, `<root_dir0>`, and the
right suite, `<root_dir1>`. This is most commonly used to compare the output of
a test run (right suite) against the stored baselines for a test (left suite).

It proceeds as follows:

1. For both test root directories, gather the list of tests that were run.

   **Note:** Tests can be identified in two ways: by their (unique) output
   directory, or by their name. If the project is using CTest to run tests, it
   is recommended (but not required) that these be made
   identical[&dagger;](#dagger2). In this case, the CMake file that consumes
   Base should include the line

   ```cmake
   set(BASE_TEST_CTEST_NAMES_ARE_OUTPUT_PATHS ON)
   ```

   to provide extra functionality.

   <a name='dagger2'></a>
   &dagger; It may not be possible to do this for test frameworks like
   googletest, which has strong restrictions on the naming of tests.

   <details>
      <summary>Details</summary><br>

      Collect a list of all the sub-directories in `<root_dir>` that contain the
      file `report.txt`. This forms a list of all tests (referenced by output
      directory) that were run that produced checksums.

      If the CTest log is present at the location `<root_dir>/ctest.log`:

      * `BASE_TEST_CTEST_NAMES_ARE_OUTPUT_PATHS` is `ON`: Parse the CTest log to
        retrieve the list of tests that were run. If this is different to the
        list collected above, report the differences.

      * `BASE_TEST_CTEST_NAMES_ARE_OUTPUT_PATHS` is `OFF`: Parse the CTest log
        to retrieve the number of tests that were run. If this is different to
        the size of the list collected above, report the difference.

   </details>


2. Find the common set of tests between both directories.

3. Compare the common tests using `comparison_func()`. For each compared test:

   * `output_type == COT_SHORT_DIFF`: If there are differences, write a `git
     diff`-style comparison to the right-suite directory for the test under the
     filename `baseline_short.diff`. Include only lines from the comparison that
     differ between the left and right suites;

   * `output_type == COT_FULL_DIFF`: If there are differences, write a `git
     diff`-style comparison to the right-suite directory for the test under the
     filename `baseline_full.diff`. Include all lines from the comparison, even
     those that are identical;

   * `output_type == COT_UPDATE` or `COT_MIRROR`: If there are differences,
     replace the test's report in the left suite with the report in the right
     suite.

4. If `output_type == COT_MIRROR`, remove any reports that exist in the left
   suite but not in the right suite.

5. Write a summary of the differences to a .diff file in the right suite root
   directory.

   <details>
      <summary>Details</summary>

      1. For each test, write the test name, the number of each kind of
         [`Difference`](#the-difference-class) that was found, and the path to
         the diff file;

      2. Write the number of tests for which differences were found;

      3. List the tests that only appeared in the left suite;

      4. List the tests that only appeared in the right suite.
   </details>

## Additional tools

Additional tools that may be useful in setting up a test system with Base/Test
have been added to the
[`toolbox`](https://git.autodesk.com/modeling-components/toolbox/) repository
(in the `test/` directory).

### test_compare.py

This script can be used to easily compare outputs between different test runs.
It makes use of the report executable and therefore can be used to diff, update
or mirror test outputs.

### test_scan.py

Some test frameworks do not automatically compile a list of tests. test_scan.py
is a flexible script that can be used to scan a directory for test files and
write their relative paths into a file in a prescribed format.

# Getting started with Base/Test in your project

This brief overview will describe how to make use of the Base/Test functionality
within a Catch2-based test system for a project that is built into a consumable
library (henceforth referred to as PROJECT).

1. Create a Catch2-based test executable for PROJECT, which we shall call
   PROJECT_test. This should be linked to PROJECT and statically linked to Base.

   Ensure that all test names are valid relative paths that can be appended to
   the output root directory to determine a unique output directory for each
   test. Ensure that the line `set(BASE_TEST_CTEST_NAMES_ARE_OUTPUT_PATHS ON)`
   is included in the encompassing CMake script as described in [Producing test
   comparison reports](#producing-test-comparison-reports).

2. Create files TestEnvironment.(cc|hh) and add them to PROJECT_test. These will
   form the bridge that provides the link to Base/Test. Create the simple class
   `Environment` that inherits from the [`Paths` class](#test-io) (appropriate
   headers will need to be included):

    **TestEnvironment.hh**
    ```cpp
    namespace Test
    {

    class Environment : public Paths
    {
    public:
      Environment(const char* const _in_rel_path);
      ~Environment();
    };

    } // namespace Test
    ```

    **TestEnvironment.cc**
    ```cpp
    namespace
    {
      std::string catch_test_name()
      {
        return Catch::getResultCapture().getCurrentTestName();
      }
    }

    namespace Test
    {
    Environment::Environment(const char* const _in_rel_path)
        : Paths(<test_root_path>, _in_rel_path, <output_root_path>,
              catch_test_name().cstr())
    {
    }

    Environment::~Environment()
    {
      // Write some checksums at the end of each test
      TEST(condition, record_number());
      TEST(completion, record_end());
    }
    } // namespace Test
    ```

3. Instantiate the test environment in a test and make use of the functionality
   (appropriate headers will need to be included):

    **test/tests.cpp** (for example)
    ```cpp
    TEST_CASE("dir1/dir2/name")
    {
      Test::Environment envr("test/tests.cpp");

      /* Body of test here */
      CHECK_OUTCOME(test_call());
      /* Use other Base/Test functionality */
    }
    ```

4. Add a report executable, as described in [Analysing Test
   Results](#analysing-test-results).

**CONGRATULATIONS, your project is now using Base/Test!**

The system may be improved easily in the following ways:

  * By creating a macro to avoid having to type the relative path of the file
    when instantiating the environment. For example:

    ```cpp
    namespace Test
    {
      std::string make_path_relative(const char* const _path);
    }

    #define TEST_ENVR_FROM_FILE_PATH \
      Test::Environment envr(Test::make_path_relative(__FILE__))
    ```
    and
    ```cpp
    namespace Test
    {
    std::string make_path_relative(const char* const _path)
    {
      fs::path path(_path);
      return path.is_absolute()
                ? fs::relative(path, fs::path(root_path())).generic_string()
                : path.generic_string();
    }
    }
    ```

    could be added to TestEnvironment.hh and TestEnvironment.cc respectively.

  * By creating a macro like

    ```cpp
    #define TEST_NAME_FROM_FILE_PATH Test::make_path_relative(__FILE__)
    ```

    to prevent having to give every test a name.

  * By making use of the [argument parser](#handling-command-line-arguments)
    within the `Environment` constructor to deal with command-line arguments.
    This will require [modifying the Catch2 `main()`
    function](https://github.com/catchorg/Catch2/blob/devel/docs/own-main.md),
    possibly as follows:

    ```cpp
    #include <Base/Test/TestArgs.hh>
    #define CATCH_CONFIG_RUNNER
    #include <catch2/catch.hpp>

    int main(int argc, char* argv[])
    {
      return Catch::Session().run(Test::Args::collect(argc, argv), argv);
    }
    ```

  * By adding another argument to the `Environment` constructor to allow tests
    to set the [checksum level](#checksum-levels) they run at (the constructor
    will have to modify `Checksum::run_lvl`).

  * By adding [`TEST_OUTCOME_ADD_TYPE()`](#the-outcome-class) definitions to
    allow more outcome types to be handled by
    [`CHECK_OUTCOME()`](#outcome-macros) and related macros.
