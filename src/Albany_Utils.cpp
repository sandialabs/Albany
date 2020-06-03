//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

// Include the concrete Epetra Comm's, if needed
#if defined(ALBANY_EPETRA)
#ifdef ALBANY_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#endif

#include <cstdlib>
#include <stdexcept>
#include <time.h>

#include "MatrixMarket_Tpetra.hpp"
#include "Teuchos_TestForException.hpp"
#include "Kokkos_Macros.hpp"

// For vtune
#include <sys/types.h>
#include <unistd.h>

// For stack trace
#include <execinfo.h>
#include <cstdarg>
#include <cstdio>

namespace Albany {

void
PrintHeader(std::ostream& os)
{
  os << R"(***************************************************************)" << std::endl;
  os << R"(**  ______   __       ______   ______   __   __   __  __     **)" << std::endl;
  os << R"(** /\  __ \ /\ \     /\  == \ /\  __ \ /\ "-.\ \ /\ \_\ \    **)" << std::endl;
  os << R"(** \ \  __ \\ \ \____\ \  __< \ \  __ \\ \ \-.  \\ \____ \   **)" << std::endl;
  os << R"(**  \ \_\ \_\\ \_____\\ \_____\\ \_\ \_\\ \_\\"\_\\/\_____\  **)" << std::endl;
  os << R"(**   \/_/\/_/ \/_____/ \/_____/ \/_/\/_/ \/_/ \/_/ \/_____/  **)" << std::endl;
  os << R"(**                                                           **)" << std::endl;
  os << R"(***************************************************************)" << std::endl;
  os << R"(** Trilinos git commit id - )" << ALBANY_TRILINOS_GIT_COMMIT_ID << std::endl;
  os << R"(** Albany git branch ------ )" << ALBANY_GIT_BRANCH << std::endl;
  os << R"(** Albany git commit id --- )" << ALBANY_GIT_COMMIT_ID << std::endl;
  os << R"(** Albany cxx compiler ---- )" << ALBANY_CXX_COMPILER_ID << " " << ALBANY_CXX_COMPILER_VERSION << std::endl;

#ifdef KOKKOS_COMPILER_CUDA_VERSION
  os << R"(** Albany cuda compiler --- Cuda )" << KOKKOS_COMPILER_CUDA_VERSION << std::endl;
#endif

  // Print start time
  time_t rawtime;
  time(&rawtime);
  struct tm* timeinfo = localtime(&rawtime);
  char buffer[80];
  strftime(buffer, 80, "%F at %T", timeinfo);
  os << R"(** Simulation start time -- )" << buffer << std::endl;
  os << R"(***************************************************************)" << std::endl;
}

int
CalculateNumberParams(const Teuchos::RCP<Teuchos::ParameterList>& problemParams)
{
  Teuchos::ParameterList& parameterParams =
      problemParams->sublist("Parameters");
  int  num_param_vecs = parameterParams.get("Number of Parameter Vectors", 0);
  bool using_old_parameter_list = false;
  if (parameterParams.isType<int>("Number")) {
    int numParameters = parameterParams.get<int>("Number");
    if (numParameters > 0) {
      num_param_vecs           = 1;
      using_old_parameter_list = true;
    }
  }
  int np = 0;
  for (int i = 0; i < num_param_vecs; ++i) {
    Teuchos::ParameterList& pList =
        using_old_parameter_list ?
            parameterParams :
            parameterParams.sublist(Albany::strint("Parameter Vector", i));
    np += pList.get<int>("Number");
  }
  return np; 
}
void
ReplaceDiagonalEntries(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Vector>&    diag)
{
  Teuchos::ArrayRCP<const ST> diag_constView = diag->get1dView();
  for (size_t i = 0; i < matrix->getNodeNumRows(); i++) {
    auto               NumEntries = matrix->getNumEntriesInLocalRow(i);
    Teuchos::Array<LO> Indices(NumEntries);
    Teuchos::Array<ST> Values(NumEntries);
    matrix->getLocalRowCopy(i, Indices(), Values(), NumEntries);
    GO global_row = matrix->getRowMap()->getGlobalElement(i);
    for (size_t j = 0; j < NumEntries; j++) {
      GO global_col = matrix->getColMap()->getGlobalElement(Indices[j]);
      if (global_row == global_col) {
        Teuchos::Array<ST> matrixEntriesT(1);
        Teuchos::Array<LO> matrixIndicesT(1);
        matrixEntriesT[0] = diag_constView[i];
        matrixIndicesT[0] = Indices[j];
        matrix->replaceLocalValues(i, matrixIndicesT(), matrixEntriesT());
      }
    }
  }
}

void
InvAbsRowSum(
    Teuchos::RCP<Tpetra_Vector>&         invAbsRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix)
{
  // Check that invAbsRowSumsTpetra and matrix have same map
  ALBANY_ASSERT(
      invAbsRowSumsTpetra->getMap()->isSameAs(*(matrix->getRowMap())),
      "Error in Albany::InvAbsRowSum!  "
      "Input vector must have same map as row map of input matrix!");

  invAbsRowSumsTpetra->putScalar(0.0);
  Teuchos::ArrayRCP<double> invAbsRowSumsTpetra_nonconstView =
      invAbsRowSumsTpetra->get1dViewNonConst();
  for (size_t row = 0; row < invAbsRowSumsTpetra->getLocalLength(); row++) {
    auto               numEntriesRow = matrix->getNumEntriesInLocalRow(row);
    Teuchos::Array<LO> indices(numEntriesRow);
    Teuchos::Array<ST> values(numEntriesRow);
    matrix->getLocalRowCopy(row, indices(), values(), numEntriesRow);
    ST scale = 0.0;
    for (size_t j = 0; j < numEntriesRow; j++) { scale += std::abs(values[j]); }

    if (scale < 1.0e-16) {
      invAbsRowSumsTpetra_nonconstView[row] = 0.0;
    } else {
      invAbsRowSumsTpetra_nonconstView[row] = 1.0 / scale;
    }
  }
}

void
AbsRowSum(
    Teuchos::RCP<Tpetra_Vector>&         absRowSumsTpetra,
    const Teuchos::RCP<Tpetra_CrsMatrix> matrix)
{
  // Check that absRowSumsTpetra and matrix have same map
  ALBANY_ASSERT(
      absRowSumsTpetra->getMap()->isSameAs(*(matrix->getRowMap())),
      "Error in Albany::AbsRowSum!  "
      "Input vector must have same map as row map of input matrix!");
  absRowSumsTpetra->putScalar(0.0);
  Teuchos::ArrayRCP<double> absRowSumsTpetra_nonconstView =
      absRowSumsTpetra->get1dViewNonConst();
  for (size_t row = 0; row < absRowSumsTpetra->getLocalLength(); row++) {
    auto               numEntriesRow = matrix->getNumEntriesInLocalRow(row);
    Teuchos::Array<LO> indices(numEntriesRow);
    Teuchos::Array<ST> values(numEntriesRow);
    matrix->getLocalRowCopy(row, indices(), values(), numEntriesRow);
    ST scale = 0.0;
    for (size_t j = 0; j < numEntriesRow; j++) { scale += std::abs(values[j]); }
    absRowSumsTpetra_nonconstView[row] = scale;
  }
}

std::string
strint(const std::string s, const int i, const char delim)
{
  std::ostringstream ss;
  ss << s << delim << i;
  return ss.str();
}

bool
isValidInitString(const std::string& initString)
{
  // Make sure the first part of the string has the correct verbiage
  std::string verbiage("initial value ");
  size_t      pos = initString.find(verbiage);
  if (pos != 0) return false;

  // Make sure the rest of the string has only allowable characters
  std::string valueString =
      initString.substr(verbiage.size(), initString.size() - verbiage.size());
  for (std::string::iterator it = valueString.begin(); it != valueString.end();
       it++) {
    std::string charAsString(1, *it);
    pos = charAsString.find_first_of("0123456789.-+eE");
    if (pos == std::string::npos) return false;
  }

  return true;
}

std::string
doubleToInitString(double val)
{
  std::string       verbiage("initial value ");
  std::stringstream ss;
  ss << verbiage << val;
  return ss.str();
}

double
initStringToDouble(const std::string& initString)
{
  ALBANY_ASSERT(
      isValidInitString(initString),
      " initStringToDouble() called with invalid initialization string: "
          << initString);
  std::string verbiage("initial value ");
  std::string valueString =
      initString.substr(verbiage.size(), initString.size() - verbiage.size());
  return std::atof(valueString.c_str());
}

void
splitStringOnDelim(
    const std::string&        s,
    char                      delim,
    std::vector<std::string>& elems)
{
  std::stringstream ss(s);
  std::string       item;
  while (std::getline(ss, item, delim)) { elems.push_back(item); }
}

std::string
getFileExtension(std::string const& filename)
{
  auto const pos = filename.find_last_of(".");
  return filename.substr(pos + 1);
}

void
printThyraVector(std::ostream& os, const Teuchos::RCP<const Thyra_Vector>& vec)
{
  Teuchos::ArrayRCP<const ST> vv          = Albany::getLocalData(vec);
  const int                   localLength = vv.size();

  os << std::setw(10) << std::endl;
  for (int i = 0; i < localLength; ++i) {
    os.width(20);
    os << "             " << std::left << vv[i] << std::endl;
  }
}

void
printThyraVector(
    std::ostream&                           os,
    const Teuchos::Array<std::string>&      names,
    const Teuchos::RCP<const Thyra_Vector>& vec)
{
  Teuchos::ArrayRCP<const ST> vv          = Albany::getLocalData(vec);
  const int                   localLength = vv.size();

  TEUCHOS_TEST_FOR_EXCEPTION(
      names.size() != localLength,
      std::logic_error,
      "Error! names and mvec length do not match.\n");

  os << std::setw(10) << std::endl;
  for (int i = 0; i < localLength; ++i) {
    os.width(20);
    os << "   " << std::left << names[i] << "\t" << vv[i] << std::endl;
  }
}

void
printThyraMultiVector(
    std::ostream&                                                    os,
    const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>>& names,
    const Teuchos::RCP<const Thyra_MultiVector>&                     mvec)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> mvv =
      Albany::getLocalData(mvec);
  const int numVecs     = mvec->domain()->dim();
  const int localLength = mvv.size() > 0 ? mvv[0].size() : 0;
  TEUCHOS_TEST_FOR_EXCEPTION(
      names.size() != localLength,
      std::logic_error,
      "Error! names and mvec length do not match.\n");

  os << std::setw(10) << std::endl;
  for (int row = 0; row < localLength; ++row) {
    for (int col = 0; col < numVecs; ++col) {
      os.width(20);
      os << "   " << std::left << (*names[col])[row] << "\t" << mvv[col][row]
         << std::endl;
    }
    os << std::endl;
  }
}

void
printThyraMultiVector(
    std::ostream&                                os,
    const Teuchos::RCP<const Thyra_MultiVector>& mvec)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> mvv =
      Albany::getLocalData(mvec);

  const int numVecs     = mvec->domain()->dim();
  const int localLength = mvv.size() > 0 ? mvv[0].size() : 0;
  os << std::setw(10) << std::endl;
  for (int row = 0; row < localLength; ++row) {
    for (int col = 0; col < numVecs; ++col) {
      os.width(20);
      os << "             " << std::left << mvv[col][row];
    }
    os << std::endl;
  }
}

//
//
//
template <>
void
writeMatrixMarket<const Tpetra_Map>(
    const Teuchos::RCP<const Tpetra_Map>& map,
    const std::string&                    prefix,
    int const                             counter)
{
  if (map.is_null()) { return; }

  std::ostringstream oss;
  oss << prefix;
  if (counter >= 0) {
    oss << '-' << std::setfill('0') << std::setw(3) << counter;
  }
  oss << ".mm";

  const std::string& filename = oss.str();

  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeMapFile(filename, *map);
}

//
//
//
template <>
void
writeMatrixMarket<const Tpetra_Vector>(
    const Teuchos::RCP<const Tpetra_Vector>& v,
    const std::string&                       prefix,
    int const                                counter)
{
  if (v.is_null()) { return; }

  std::ostringstream oss;

  oss << prefix;
  if (counter >= 0) {
    oss << '-' << std::setfill('0') << std::setw(3) << counter;
  }
  oss << ".mm";

  const std::string& filename = oss.str();

  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(filename, v);
}

//
//
//
template <>
void
writeMatrixMarket<const Tpetra_MultiVector>(
    const Teuchos::RCP<const Tpetra_MultiVector>& mv,
    const std::string&                            prefix,
    int const                                     counter)
{
  if (mv.is_null()) { return; }

  std::ostringstream oss;

  oss << prefix;
  if (counter >= 0) {
    oss << '-' << std::setfill('0') << std::setw(3) << counter;
  }
  oss << ".mm";

  const std::string& filename = oss.str();

  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(filename, mv);
}

//
//
//
template <>
void
writeMatrixMarket<const Tpetra_CrsMatrix>(
    const Teuchos::RCP<const Tpetra_CrsMatrix>& A,
    const std::string&                          prefix,
    int const                                   counter)
{
  if (A.is_null()) { return; }

  std::ostringstream oss;

  oss << prefix;
  if (counter >= 0) {
    oss << '-' << std::setfill('0') << std::setw(3) << counter;
  }
  oss << ".mm";

  const std::string& filename = oss.str();

  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(filename, A);
}

CmdLineArgs::CmdLineArgs(
    const std::string& default_yaml_filename,
    const std::string& default_yaml_filename2,
    const std::string& default_yaml_filename3)
    : yaml_filename(default_yaml_filename),
      yaml_filename2(default_yaml_filename2),
      yaml_filename3(default_yaml_filename3),
      has_first_yaml_file(false),
      has_second_yaml_file(false),
      has_third_yaml_file(false),
      vtune(false)
{
}

void
CmdLineArgs::parse_cmdline(int argc, char** argv, std::ostream& os)
{
  bool found_first_yaml_file  = false;
  bool found_second_yaml_file = false;
  for (int arg = 1; arg < argc; ++arg) {
    if (!std::strcmp(argv[arg], "--help")) {
      os << argv[0]
         << " [--vtune] [inputfile1.yaml] [inputfile2.yaml] "
            "[inputfile3.yaml]\n";
      std::exit(1);
    } else if (!std::strcmp(argv[arg], "--vtune")) {
      vtune = true;
    } else {
      if (!found_first_yaml_file) {
        yaml_filename         = argv[arg];
        found_first_yaml_file = true;
        has_first_yaml_file   = true;
      } else if (!found_second_yaml_file) {
        yaml_filename2         = argv[arg];
        found_second_yaml_file = true;
        has_second_yaml_file   = true;
      } else {
        yaml_filename3      = argv[arg];
        has_third_yaml_file = true;
      }
    }
  }
}

void
connect_vtune(const int p_rank)
{
  std::stringstream cmd;
  pid_t             my_os_pid  = getpid();
  const std::string vtune_loc  = "amplxe-cl";
  const std::string output_dir = "./vtune/vtune.";
  cmd << vtune_loc << " -collect hotspots -result-dir " << output_dir << p_rank
      << " -target-pid " << my_os_pid << " &";
  if (p_rank == 0) std::cout << cmd.str() << std::endl;
  safe_system(cmd.str().c_str());
  safe_system("sleep 10");
}

void
do_stack_trace()
{
  void*  callstack[128];
  int    i, frames = backtrace(callstack, 128);
  char** strs = backtrace_symbols(callstack, frames);
  for (i = 0; i < frames; ++i) { printf("%s\n", strs[i]); }
  free(strs);
}

void
safe_fscanf(int nitems, FILE* file, const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  int ret = vfscanf(file, format, ap);
  va_end(ap);
  ALBANY_ASSERT(
      ret == nitems,
      ret << "=safe_fscanf(" << nitems << ", " << file << ", \"" << format
          << "\")");
}

void
safe_sscanf(int nitems, const char* str, const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  int ret = vsscanf(str, format, ap);
  va_end(ap);
  ALBANY_ASSERT(
      ret == nitems,
      ret << "=safe_sscanf(" << nitems << ", \"" << str << "\", \"" << format
          << "\")");
}

void
safe_fgets(char* str, int size, FILE* stream)
{
  char* ret = fgets(str, size, stream);
  ALBANY_ASSERT(
      ret == str,
      ret << "=safe_fgets(" << static_cast<void*>(str) << ", " << size << ", "
          << stream << ")");
}

void
safe_system(char const* str)
{
  ALBANY_ASSERT(str, "safe_system called with null command string\n");
  int ret = system(str);
  ALBANY_ASSERT(str, ret << "=safe_system(\"" << str << "\")");
}

void
assert_fail(std::string const& msg)
{
  std::cerr << msg;
  abort();
}

BuildType
build_type(const BuildType value)
{
  // Recall how static local variable work: the following are created (and
  // initialized) only once
  static bool      initialized_ = false;
  static BuildType value_       = BuildType::None;
  if (!initialized_ && (value != BuildType::None)) {
    value_       = value;
    initialized_ = true;
  }
  return value_;
}

//
//
//
int
getProcRank()
{
  int rank{0};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

}  // namespace Albany
