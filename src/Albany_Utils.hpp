//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UTILS_H
#define ALBANY_UTILS_H

// Get Albany configuration macros
#include "Albany_config.h"

// For cudaCheckError
#include <stdexcept>

#include <sstream>

#include "Albany_DataTypes.hpp"
#include "Teuchos_RCP.hpp"

#if defined(ALBANY_EPETRA)
#include "Epetra_Comm.h"
#endif

// Checks if the previous Kokkos::Cuda kernel has failed
#ifdef ALBANY_CUDA_ERROR_CHECK
#define cudaCheckError() \
  { cudaError(__FILE__, __LINE__); }
inline void
cudaError(const char* file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(
        stderr, "CUDA Error: %s before %s:%d\n", cudaGetErrorString(err), file,
        line);
    throw std::runtime_error(cudaGetErrorString(err));
  }
}
#else
#define cudaCheckError()
#endif

// NVTX Range creates a colored range which can be viewed on the nvvp timeline
// (from Parallel Forall blog)
#ifdef ALBANY_CUDA_NVTX
#include "nvToolsExt.h"
static const uint32_t nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00,
    0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_nvtx_colors = sizeof(nvtx_colors)/sizeof(uint32_t);
#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_nvtx_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = nvtx_colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

namespace Albany {

// Helper function which replaces the diagonal of a matrix
void
ReplaceDiagonalEntries(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Vector>& diag);

// Helper function which computes absolute values of the rowsum
// of a matrix, takes its inverse, and puts it in a vector.
void
InvAbsRowSum(
    Teuchos::RCP<Tpetra_Vector>& invAbsRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix);

// Helper function which computes absolute values of the rowsum
// of a matrix, and puts it in a vector.
void
AbsRowSum(
    Teuchos::RCP<Tpetra_Vector>& absRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix);

#if defined(ALBANY_EPETRA)

Albany_MPI_Comm
getMpiCommFromEpetraComm(const Epetra_Comm& ec);

Albany_MPI_Comm
getMpiCommFromEpetraComm(Epetra_Comm& ec);
Teuchos::RCP<Epetra_Comm>
createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc);
Teuchos::RCP<Epetra_Comm>
createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc);
Teuchos::RCP<Teuchos_Comm>
createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec);
Teuchos::RCP<Teuchos_Comm>
createTeuchosCommFromEpetraComm(const Epetra_Comm& ec);

#endif

// Helper function which replaces the diagonal of a matrix
void
ReplaceDiagonalEntries(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Vector>& diag);

// Helper function which creates diagonal vector with entries equal to the
// absolute value of the rowsum of a matrix.

Albany_MPI_Comm
getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc);

Teuchos::RCP<Teuchos_Comm>
createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc);

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
    const std::string& s, char delim, std::vector<std::string>& elems);

/// Get file name extension
std::string
getFileExtension(std::string const& filename);

//! Nicely prints out a Tpetra Vector
void
printTpetraVector(
    std::ostream& os, const Teuchos::RCP<const Tpetra_Vector>& vec);
void
printTpetraVector(
    std::ostream& os, const Teuchos::Array<std::string>& names,
    const Teuchos::RCP<const Tpetra_Vector>& vec);

//! Nicely prints out a Tpetra MultiVector
void
printTpetraVector(
    std::ostream& os, const Teuchos::RCP<const Tpetra_MultiVector>& vec);
void
printTpetraVector(
    std::ostream& os,
    const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>>& names,
    const Teuchos::RCP<const Tpetra_MultiVector>& vec);

/// Write to matrix market format a vector, matrix or map.
template<typename LinearAlgebraObjectType>
void
writeMatrixMarket(
    const Teuchos::RCP<const LinearAlgebraObjectType>& A,
    const std::string& prefix,
    int const counter = -1);

template<typename LinearAlgebraObjectType>
void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<LinearAlgebraObjectType>>& x,
    const std::string& prefix,
    int const counter = -1)
{
  for (auto i = 0; i < x.size(); ++i) {
    std::ostringstream oss;

    oss << prefix << '-' << std::setfill('0') << std::setw(2) << i;

    const std::string& new_prefix = oss.str();

    writeMatrixMarket(x[i].getConst(), new_prefix, counter);
  }
}
/////

//void
//writeMatrixMarket(
//    Teuchos::RCP<Tpetra_MultiVector const> const& x, std::string const& prefix,
//    int const counter = -1);

//void
//writeMatrixMarket(
//    Teuchos::RCP<Tpetra_CrsMatrix const> const& A, std::string const& prefix,
//    int const counter = -1);

//void
//writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> const& x,
//    std::string const& prefix, int const counter = -1);

//void
//writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix const>> const& A,
//    std::string const& prefix, int const counter = -1);

//void
//writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_Vector>> const& x,
//    std::string const& prefix, int const counter = -1);

//void
//writeMatrixMarket(
//    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> const& A,
//    std::string const& prefix, int const counter = -1);

// Parses and stores command-line arguments
struct CmdLineArgs {
  std::string xml_filename;
  std::string xml_filename2;
  std::string xml_filename3;
  bool has_first_xml_file;
  bool has_second_xml_file;
  bool has_third_xml_file;
  bool vtune;

  CmdLineArgs(
      const std::string& default_xml_filename = "input.xml",
      const std::string& default_xml_filename2 = "",
      const std::string& default_xml_filename3 = "");
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
enum class BuildType {None, Tpetra, Epetra};
BuildType build_type(const BuildType value=BuildType::None);

}  // end namespace Albany

#ifdef __CUDA_ARCH__
#define ALBANY_ASSERT_IMPL(cond, ...) assert(cond)
#else
#define ALBANY_ASSERT_IMPL(cond, msg, ...)          \
  do {                                              \
    if (!(cond)) {                                  \
      std::ostringstream omsg;                      \
      omsg << #cond " failed at ";                  \
      omsg << __FILE__ << " +" << __LINE__ << '\n'; \
      omsg << msg << '\n';                          \
      Albany::assert_fail(omsg.str());              \
    }                                               \
  } while (0)
#endif

#define ALBANY_ASSERT(...) ALBANY_ASSERT_IMPL(__VA_ARGS__, "")

#ifdef NDEBUG
#define ALBANY_EXPECT(...)
#else
#define ALBANY_EXPECT(...) ALBANY_ASSERT(__VA_ARGS__)
#endif

#define ALBANY_ALWAYS_ASSERT(cond) ALBANY_ASSERT(cond)
#define ALBANY_ALWAYS_ASSERT_VERBOSE(cond, msg) ALBANY_ASSERT(cond, msg)
#define ALBANY_DEBUG_ASSERT(cond) ALBANY_EXPECT(cond)
#define ALBANY_DEBUG_ASSERT_VERBOSE(cond, msg) ALBANY_EXPECT(cond, msg)

#endif  // ALBANY_UTILS
