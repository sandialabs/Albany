//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UTILS_H
#define ALBANY_UTILS_H

// Get Albany configuration macros
#include "Albany_config.h"

#include <sstream>

#include "Albany_CommUtils.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

//! Print ascii art and version information
void
PrintHeader(std::ostream& os);

//! Helper function to calculate the number of parameters in a problem
int
CalculateNumberParams(const Teuchos::RCP<Teuchos::ParameterList>& problemParams); 

// Helper function which replaces the diagonal of a matrix
void
ReplaceDiagonalEntries(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Vector>&    diag);

// Helper function which computes absolute values of the rowsum
// of a matrix, takes its inverse, and puts it in a vector.
void
InvAbsRowSum(
    Teuchos::RCP<Tpetra_Vector>&         invAbsRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix);

// Helper function which computes absolute values of the rowsum
// of a matrix, and puts it in a vector.
void
AbsRowSum(
    Teuchos::RCP<Tpetra_Vector>&         absRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix);

// Helper function which replaces the diagonal of a matrix
void
ReplaceDiagonalEntries(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Vector>&    diag);

//! Utility to make a string out of a string + int with a delimiter:
//! strint("dog",2,' ') = "dog 2"
//! The default delimiter is ' '. Potential delimiters include '_' - "dog_2"
std::string
strint(const std::string s, const int i, const char delim = ' ');

//! Returns true of the given string is a valid initialization string of the
//! format "initial value 1.54"
bool
isValidInitString(const std::string& initString);

//! Converts a double to an initialization string:  doubleToInitString(1.54) =
//! "initial value 1.54"
std::string
doubleToInitString(double val);

//! Converts an init string to a double:  initStringToDouble("initial value
//! 1.54") = 1.54
double
initStringToDouble(const std::string& initString);

//! Splits a std::string on a delimiter
void
splitStringOnDelim(
    const std::string&        s,
    char                      delim,
    std::vector<std::string>& elems);

/// Get file name extension
std::string
getFileExtension(std::string const& filename);

//! Nicely prints out a Thyra Vector
void
printThyraVector(std::ostream& os, const Teuchos::RCP<const Thyra_Vector>& vec);
void
printThyraVector(
    std::ostream&                           os,
    const Teuchos::Array<std::string>&      names,
    const Teuchos::RCP<const Thyra_Vector>& vec);

//! Inlined product version
inline void
printThyraVector(
    std::ostream&                                  os,
    const Teuchos::RCP<const Thyra_ProductVector>& vec)
{
  for (int i = 0; i < vec->productSpace()->numBlocks(); ++i) {
    printThyraVector(os, vec->getVectorBlock(i));
  }
}

//! Nicely prints out a Thyra MultiVector
void
printThyraMultiVector(
    std::ostream&                                os,
    const Teuchos::RCP<const Thyra_MultiVector>& vec);
void
printThyraMultiVector(
    std::ostream&                                                    os,
    const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>>& names,
    const Teuchos::RCP<const Thyra_MultiVector>&                     vec);

//! Inlined product version
inline void
printThyraVector(
    std::ostream&                                       os,
    const Teuchos::RCP<const Thyra_ProductMultiVector>& vec)
{
  for (int i = 0; i < vec->productSpace()->numBlocks(); ++i) {
    printThyraMultiVector(os, vec->getMultiVectorBlock(i));
  }
}

/// Write to matrix market format a vector, matrix or map.
template <typename LinearAlgebraObjectType>
void
writeMatrixMarket(
    const Teuchos::RCP<LinearAlgebraObjectType>& A,
    const std::string&                           prefix,
    int const                                    counter = -1);

template <typename LinearAlgebraObjectType>
void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<LinearAlgebraObjectType>>& x,
    const std::string&                                           prefix,
    int const                                                    counter = -1)
{
  for (auto i = 0; i < x.size(); ++i) {
    std::ostringstream oss;

    oss << prefix << '-' << std::setfill('0') << std::setw(2) << i;

    const std::string& new_prefix = oss.str();

    writeMatrixMarket(x[i].getConst(), new_prefix, counter);
  }
}
/////

// void
// writeMatrixMarket(
//    Teuchos::RCP<Tpetra_MultiVector const> const& x, std::string const&
//    prefix, int const counter = -1);

// void
// writeMatrixMarket(
//    Teuchos::RCP<Tpetra_CrsMatrix const> const& A, std::string const& prefix,
//    int const counter = -1);

// void
// writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> const& x,
//    std::string const& prefix, int const counter = -1);

// void
// writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix const>> const& A,
//    std::string const& prefix, int const counter = -1);

// void
// writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_Vector>> const& x,
//    std::string const& prefix, int const counter = -1);

// void
// writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> const& A,
//    std::string const& prefix, int const counter = -1);

// Parses and stores command-line arguments
struct CmdLineArgs
{
  std::string yaml_filename;
  std::string yaml_filename2;
  std::string yaml_filename3;
  bool        has_first_yaml_file;
  bool        has_second_yaml_file;
  bool        has_third_yaml_file;
  bool        vtune;

  CmdLineArgs(
      const std::string& default_yaml_filename  = "input.yaml",
      const std::string& default_yaml_filename2 = "",
      const std::string& default_yaml_filename3 = "");
  void
  parse_cmdline(int argc, char** argv, std::ostream& os);
};

// Connect executable to vtune for profiling
void
connect_vtune(const int p_rank);

// Do a nice stack trace for debugging
void
do_stack_trace();

// Check returns codes and throw Teuchos exceptions
// Useful for silencing compiler warnings about unused return codes
void
safe_fscanf(int nitems, FILE* file, const char* format, ...);
void
safe_sscanf(int nitems, const char* str, const char* format, ...);
void
safe_fgets(char* str, int size, FILE* stream);
void
safe_system(char const* str);

void
assert_fail(std::string const& msg) __attribute__((noreturn));

/// \brief Get/Set the Albany build type
///
/// \params value [in] The Albany build type to set
///
/// \notes This function acts as both a getter and setter for the Albany
/// build type.  The *first* time the optional param \c value is passed to
/// this function, it set as the build type.  The build type is always returned.
///
/// For executables/problems that need to know the build type, this function
/// must be called early in the main function to set the appropriate build type,
/// preferably as one of the first actions of the program (before initializing
/// MPI)..
///
/// This function and capablity may not be necessary after the transition away
/// from Epetra is complete.  In the meantime, it is used in a handful of places
/// to execute code that is specific to either Tpetra or Epetra.
enum class BuildType
{
  None,
  Tpetra,
  Epetra
};
BuildType
build_type(const BuildType value = BuildType::None);

//
//
//
int
getProcRank();

}  // end namespace Albany

#endif  // ALBANY_UTILS
