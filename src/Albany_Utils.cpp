//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GitVersion.h"
#include "Albany_StringUtils.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <time.h>

#include "MatrixMarket_Tpetra.hpp"
#include "Teuchos_TestForException.hpp"
#include "Kokkos_Macros.hpp"

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
  os << R"(** Albany cxx compiler ---- )" << CMAKE_CXX_COMPILER_ID << " " << CMAKE_CXX_COMPILER_VERSION << std::endl;

#ifdef KOKKOS_COMPILER_CUDA_VERSION
  os << R"(** Albany cuda compiler --- Cuda )" << KOKKOS_COMPILER_CUDA_VERSION << std::endl;
#endif

  // Print fad types
#if defined(ALBANY_FAD_TYPE_SFAD)
  os << R"(** Albany FadType --------- SFad)" << ALBANY_SFAD_SIZE << std::endl;
#elif defined(ALBANY_FAD_TYPE_SLFAD)
  os << R"(** Albany FadType --------- SLFad)" << ALBANY_SLFAD_SIZE << std::endl;
#else
  os << R"(** Albany FadType --------- DFad)" << std::endl;
#endif
#if defined(ALBANY_TAN_FAD_TYPE_SFAD)
  os << R"(** Albany TanFadType ------ SFad)" << ALBANY_TAN_SFAD_SIZE << std::endl;
#elif defined(ALBANY_TAN_FAD_TYPE_SLFAD)
  os << R"(** Albany TanFadType ------ SLFad)" << ALBANY_TAN_SLFAD_SIZE << std::endl;
#else
  os << R"(** Albany TanFadType ------ DFad)" << std::endl;
#endif
#if defined(ALBANY_HES_VEC_FAD_TYPE_SFAD)
  os << R"(** Albany HessianVecFad  -- SFad)" << ALBANY_HES_VEC_SFAD_SIZE << std::endl;
#elif defined(ALBANY_HES_VEC_FAD_TYPE_SLFAD)
  os << R"(** Albany HessianVecFad  -- SLFad)" << ALBANY_HES_VEC_SLFAD_SIZE << std::endl;
#else
  os << R"(** Albany HessianVecFad  -- DFad)" << std::endl;
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

void
PrintMPIInfo(std::ostream& os)
{
  const auto comm = Albany::getDefaultComm();
  const auto rank = comm->getRank();
  const auto size = comm->getSize();
  int nameLen;
  char procName[MPI_MAX_PROCESSOR_NAME];
  ::MPI_Get_processor_name(procName, &nameLen);
  std::ostringstream oss;
  oss << "Rank " << rank << " of " << size << " exists on processor " << procName << std::endl;
  os << oss.str() << std::flush;
  comm->barrier();
}

int
CalculateNumberParams(const Teuchos::RCP<const Teuchos::ParameterList> problemParams, int * numScalarParams, int * numDistributedParams)
{
  const Teuchos::ParameterList& parameterParams =
      problemParams->sublist("Parameters");
  int nsp(0), ndp(0);
  if(parameterParams.isParameter("Number Of Parameters")) {
    int  num_param_vecs = parameterParams.get<int>("Number Of Parameters");
    for (int i = 0; i < num_param_vecs; ++i) {
      const Teuchos::ParameterList& pList =
          parameterParams.sublist(util::strint("Parameter", i));
      const std::string& parameterType = pList.isParameter("Type") ?
          pList.get<std::string>("Type") : std::string("Scalar");
      if(parameterType == "Scalar")
        ++nsp;
      else if (parameterType == "Vector")
        nsp += pList.get<int>("Dimension");
      else if (parameterType == "Distributed")
        ++ndp;
      else
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            Teuchos::Exceptions::InvalidParameter,
            std::endl
                << "Error!  In Albany::CalculateNumberParams:  "
                << "Parameter vector "
                << i
                << " is of the type: \""
                << parameterType
                << "\"; this type is unsupported.\n"
                << "Please use a valid type: \"Scalar\", \"Vector\", or \"Distributed\"."
                << std::endl);
    }
  }

  if(numScalarParams != NULL)
    *numScalarParams = nsp;

  if(numDistributedParams != NULL)
    *numDistributedParams = ndp; 

  return nsp+ndp; 
}

void
getParameterSizes(const Teuchos::ParameterList parameterParams, int &total_num_param_vecs, int &num_param_vecs, int &num_dist_param_vecs ) {
  total_num_param_vecs = 0;
  num_param_vecs = 0;
  num_dist_param_vecs = 0;
  total_num_param_vecs = parameterParams.get<int>("Number Of Parameters");
  bool previous_param_is_distributed = false;

  for (int l = 0; l < total_num_param_vecs; ++l) {
    const Teuchos::ParameterList& pList =
        parameterParams.sublist(util::strint("Parameter", l));

    const std::string parameterType = pList.isParameter("Type") ?
        pList.get<std::string>("Type") : std::string("Scalar");
    if(parameterType == "Scalar" || parameterType == "Vector") {
      TEUCHOS_TEST_FOR_EXCEPTION(
          previous_param_is_distributed,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  In Albany::getParameterSizes:  "
              << "Parameter vector "
              << l
              << " is not distributed and the parameter "
              << l-1
              << " was distributed; please reorder the parameters swapping them.\n"
              << "All non-distributed parameters (\"Scalar\" and \"Vector\") must be listed before the distributed parameters"
              << std::endl);
      ++num_param_vecs;
    }
    else if (parameterType =="Distributed") {
      ++num_dist_param_vecs;
      previous_param_is_distributed = true;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  In Albany::getParameterSizes:  "
              << "Parameter vector "
              << l
              << " is of the type: \""
              << parameterType
              << "\"; this type is unsupported.\n"
              << "Please use a valid type: \"Scalar\", \"Vector\", or \"Distributed\"."
              << std::endl);
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
  using indices_type = typename Tpetra_CrsMatrix::local_inds_host_view_type;
  using values_type  = typename Tpetra_CrsMatrix::values_host_view_type;
  for (size_t row = 0; row < invAbsRowSumsTpetra->getLocalLength(); row++) {
    indices_type indices;
    values_type  values;
    matrix->getLocalRowView(row, indices, values);
    ST scale = 0.0;
    for (size_t j = 0; j < indices.size(); j++) {
      scale += std::abs(values[j]);
    }

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

  using indices_type = typename Tpetra_CrsMatrix::local_inds_host_view_type;
  using values_type  = typename Tpetra_CrsMatrix::values_host_view_type;
  for (size_t row = 0; row < absRowSumsTpetra->getLocalLength(); row++) {
    indices_type indices;
    values_type  values;
    matrix->getLocalRowView(row, indices, values);
    ST scale = 0.0;
    for (size_t j = 0; j < values.size(); j++) {
      scale += std::abs(values[j]);
    }
    absRowSumsTpetra_nonconstView[row] = scale;
  }
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "Writing Matrix Market file " << filename << " ..." << std::endl;
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "Writing Matrix Market file " << filename << " ..." << std::endl;
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "Writing Matrix Market file " << filename << " ..." << std::endl;
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "Writing Matrix Market file " << filename << " ..." << std::endl;
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(filename, A);
}

//
//
//
template <>
void
writeMatrixMarket<const Tpetra_CrsGraph>(
    const Teuchos::RCP<const Tpetra_CrsGraph>& A,
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

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "Writing Matrix Market file " << filename << " ..." << std::endl;
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseGraphFile(filename, A);
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
      has_third_yaml_file(false)
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
         << " [inputfile1.yaml] [inputfile2.yaml] "
            "[inputfile3.yaml]\n";
      std::exit(1);
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
assert_fail(std::string const& msg)
{
  std::cerr << msg;
  abort();
}

int
getProcRank()
{
  int rank{0};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

std::string
getDebugFileName(const std::string& prefix, const std::string& suffix)
{
  int rank = getProcRank();
  std::stringstream ss;
  ss << prefix << "_rank" << rank << suffix;
  return ss.str();
}

std::ofstream&
getDebugStream(const std::string& prefix)
{
  // Static map to hold one ofstream per unique prefix
  static std::map<std::string, std::unique_ptr<std::ofstream>> debugStreams;

  // Check if stream already exists for this prefix
  auto it = debugStreams.find(prefix);
  if (it == debugStreams.end()) {
    // Create new stream with rank-specific filename
    std::string filename = getDebugFileName(prefix, ".txt");
    auto stream = std::make_unique<std::ofstream>(filename, std::ios::out);

    TEUCHOS_TEST_FOR_EXCEPTION(
      !stream->is_open(),
      std::runtime_error,
      "Error! Cannot open debug file '" << filename << "' for writing.\n");

    // Insert into map and get iterator to the inserted element
    auto result = debugStreams.emplace(prefix, std::move(stream));
    it = result.first;
  }

  return *(it->second);
}

void
flushDebugStreams()
{
  // Note: We cannot access the static map from getDebugStream here,
  // so this is a no-op. Users should call flush() on individual streams.
  // This function is provided for API completeness and future expansion.
}

}  // namespace Albany
