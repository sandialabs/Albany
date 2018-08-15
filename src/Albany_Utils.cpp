//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

// Include the concrete Epetra Comm's, if needed
#if defined(ALBANY_EPETRA)
  #ifdef ALBANY_MPI
    #include "Epetra_MpiComm.h"
  #else
    #include "Epetra_SerialComm.h"
  #endif
#endif

#include "MatrixMarket_Tpetra.hpp"
#include "Teuchos_TestForException.hpp"
#include <cstdlib>
#include <stdexcept>

// For vtune
#include <sys/types.h>
#include <unistd.h>

// For stack trace
#include <execinfo.h>
#include <cstdio>
#include <cstdarg>

namespace Albany
{

// Start of Utils to do with Communicators
#ifdef ALBANY_MPI

  void
  ReplaceDiagonalEntries(const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
                                 const Teuchos::RCP<Tpetra_Vector>& diag) {
    Teuchos::ArrayRCP<const ST> diag_constView = diag->get1dView();
    for (auto i=0; i<matrix->getNodeNumRows(); i++) {
      auto NumEntries = matrix->getNumEntriesInLocalRow(i);
      Teuchos::Array<LO> Indices(NumEntries);
      Teuchos::Array<ST> Values(NumEntries);
      matrix->getLocalRowCopy(i, Indices(), Values(), NumEntries);
      GO global_row = matrix->getRowMap()->getGlobalElement(i);
      for (auto j=0; j<NumEntries; j++) {
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
    //Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile("prec.mm", matrix);
  }

  void 
  InvAbsRowSum(Teuchos::RCP<Tpetra_Vector>& invAbsRowSumsTpetra, const Teuchos::RCP<Tpetra_CrsMatrix> matrix) {
    //Check that invAbsRowSumsTpetra and matrix have same map 
    ALBANY_ASSERT(invAbsRowSumsTpetra->getMap()->isSameAs(*(matrix->getRowMap())), 
			 "Error in Albany::InvAbsRowSum!  "
			 "Input vector must have same map as row map of input matrix!");
    invAbsRowSumsTpetra->putScalar(0.0);
    Teuchos::ArrayRCP<double> invAbsRowSumsTpetra_nonconstView = invAbsRowSumsTpetra->get1dViewNonConst(); 
    for (auto row=0; row<invAbsRowSumsTpetra->getLocalLength(); row++) {
      auto numEntriesRow = matrix->getNumEntriesInLocalRow(row); 
      Teuchos::Array<LO> indices(numEntriesRow); 
      Teuchos::Array<ST> values(numEntriesRow); 
      matrix->getLocalRowCopy(row, indices(), values(), numEntriesRow);
      ST scale = 0.0; 
      for (auto j=0; j < numEntriesRow; j++) scale += std::abs(values[j]);
      if (scale < 1.0e-16) invAbsRowSumsTpetra_nonconstView[row] = 0.0; 
      else invAbsRowSumsTpetra_nonconstView[row] = 1.0/scale; 
    }
  }

  void 
  AbsRowSum(Teuchos::RCP<Tpetra_Vector>& absRowSumsTpetra, const Teuchos::RCP<Tpetra_CrsMatrix> matrix) {
    //Check that absRowSumsTpetra and matrix have same map 
    ALBANY_ASSERT(absRowSumsTpetra->getMap()->isSameAs(*(matrix->getRowMap())), 
			 "Error in Albany::AbsRowSum!  "
			 "Input vector must have same map as row map of input matrix!");
    absRowSumsTpetra->putScalar(0.0);
    Teuchos::ArrayRCP<double> absRowSumsTpetra_nonconstView = absRowSumsTpetra->get1dViewNonConst(); 
    for (auto row=0; row<absRowSumsTpetra->getLocalLength(); row++) {
      auto numEntriesRow = matrix->getNumEntriesInLocalRow(row); 
      Teuchos::Array<LO> indices(numEntriesRow); 
      Teuchos::Array<ST> values(numEntriesRow); 
      matrix->getLocalRowCopy(row, indices(), values(), numEntriesRow);
      ST scale = 0.0; 
      for (auto j=0; j < numEntriesRow; j++) scale += std::abs(values[j]);
      absRowSumsTpetra_nonconstView[row] = scale; 
    }
  }



#if defined(ALBANY_EPETRA)  
  Albany_MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec) {
    const Epetra_MpiComm& emc = dynamic_cast<const Epetra_MpiComm&>(ec);
    return emc.Comm();
  }

  Albany_MPI_Comm getMpiCommFromEpetraComm(Epetra_Comm& ec) {
    Epetra_MpiComm& emc = dynamic_cast<Epetra_MpiComm&>(ec);
    return emc.Comm();
  }

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Epetra_MpiComm(mc));
  }

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc) {
    const Teuchos::Ptr<const Teuchos::MpiComm<int> > mpiComm =
               Teuchos::ptr_dynamic_cast<const Teuchos::MpiComm<int> >(Teuchos::ptrFromRef(*tc));
    return  createEpetraCommFromMpiComm(*mpiComm->getRawMpiComm()());
  }

  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec) {
    const Teuchos::Ptr<const Epetra_MpiComm> mpiComm =
               Teuchos::ptr_dynamic_cast<const Epetra_MpiComm>(Teuchos::ptrFromRef(*ec));
    return  createTeuchosCommFromMpiComm(mpiComm->Comm());
  }

  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromEpetraComm(const Epetra_Comm& ec) {
    const Epetra_MpiComm *mpiComm =
               dynamic_cast<const Epetra_MpiComm *>(&ec);
    return  createTeuchosCommFromMpiComm(mpiComm->Comm());
  }
#endif


  Albany_MPI_Comm getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc) {
    Teuchos::Ptr<const Teuchos::MpiComm<int> > mpiComm =
               Teuchos::ptr_dynamic_cast<const Teuchos::MpiComm<int> >(Teuchos::ptrFromRef(*tc));
    return *mpiComm->getRawMpiComm();

  }

  Teuchos::RCP<Teuchos::Comm<int> > createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(mc)));
  }

#else

#if defined(ALBANY_EPETRA)

  Albany_MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec) { return 1; }

  Albany_MPI_Comm getMpiCommFromEpetraComm(Epetra_Comm& ec) { return 1; }

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Epetra_SerialComm);
  }

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc) {
    return Teuchos::rcp(new Epetra_SerialComm);
  }

  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec) {
    return Teuchos::rcp(new Teuchos::SerialComm<int>());
  }
#endif

  Teuchos::RCP<Teuchos::Comm<int> > createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Teuchos::SerialComm<int>());
  }

#endif

  // End of Utils to do with Communicators

  std::string strint(const std::string s, const int i, const char delim) {
      std::ostringstream ss;
      ss << s << delim << i;
      return ss.str();
    }

  bool isValidInitString(const std::string& initString) {

    // Make sure the first part of the string has the correct verbiage
    std::string verbiage("initial value ");
    size_t pos = initString.find(verbiage);
    if(pos != 0)
      return false;

    // Make sure the rest of the string has only allowable characters
    std::string valueString = initString.substr(verbiage.size(), initString.size() - verbiage.size());
    int decimalPointCount = 0;
    for(std::string::iterator it=valueString.begin() ; it!=valueString.end() ; it++){
      std::string charAsString(1, *it);
      size_t pos = charAsString.find_first_of("0123456789.-+eE");
      if(pos == std::string::npos)
        return false;
    }

    return true;
  }

  std::string doubleToInitString(double val) {
    std::string verbiage("initial value ");
    std::stringstream ss;
    ss << verbiage << val;
    return ss.str();
  }

  double initStringToDouble(const std::string& initString) {
    ALBANY_ASSERT(isValidInitString(initString),
			       " initStringToDouble() called with invalid initialization string: " << initString);
    std::string verbiage("initial value ");
    std::string valueString = initString.substr(verbiage.size(), initString.size() - verbiage.size());
    return std::atof(valueString.c_str());
  }

  void splitStringOnDelim(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
  }

  std::string
  getFileExtension(std::string const & filename)
  {
    auto const pos = filename.find_last_of(".");
    return filename.substr(pos + 1);
  }

  void printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_Vector>& vec){

    Teuchos::ArrayRCP<const double> vv = vec->get1dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t i = 0; i < vec->getLocalLength(); i++){
       os.width(20);
       os << "             " << std::left << vv[i] << std::endl;
    }

  }

  void printTpetraVector(std::ostream &os, const Teuchos::Array<std::string>& names,
        const Teuchos::RCP<const Tpetra_Vector>& vec){

    Teuchos::ArrayRCP<const double> vv = vec->get1dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t i = 0; i < names.size(); i++){
       os.width(20);
//       os << "             " << std::left << vv[i] << std::endl;
       os << "   " << std::left << names[i] << "\t" << vv[i] << std::endl;
    }

  }

  void printTpetraVector(std::ostream &os, const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > >& names,
        const Teuchos::RCP<const Tpetra_MultiVector>& vec){

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double> > mvv = vec->get2dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t row = 0; row < names.size(); row++){
      for(std::size_t col = 0; col < vec->getNumVectors(); col++){
         os.width(20);
//         os << "             " << std::left << mvv[col][row] ;
         os << "   " << std::left << (*names[col])[row] << "\t" << mvv[col][row] << std::endl;
      }
      os << std::endl;
    }

  }

  void printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_MultiVector>& vec){

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double> > mvv = vec->get2dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t row = 0; row < vec->getLocalLength(); row++){
      for(std::size_t col = 0; col < vec->getNumVectors(); col++){
         os.width(20);
         os << "             " << std::left << mvv[col][row] ;
      }
      os << std::endl;
    }

  }

  //
  //
  //
  template<>
  void
  writeMatrixMarket<Tpetra_Map>(
      const Teuchos::RCP<const Tpetra_Map>& map,
      const std::string& prefix,
      int const counter)
  {
    if (map.is_null()) {
      return;
    }

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
  template<>
  void
  writeMatrixMarket<Tpetra_Vector>(
      const Teuchos::RCP<const Tpetra_Vector>& v,
      const std::string& prefix,
      int const counter)
  {
    if (v.is_null()) {
      return;
    }

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
  template<>
  void
  writeMatrixMarket<Tpetra_MultiVector>(
      const Teuchos::RCP<const Tpetra_MultiVector>& mv,
      const std::string& prefix,
      int const counter)
  {
    if (mv.is_null()) {
      return;
    }

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
  template<>
  void
  writeMatrixMarket<Tpetra_CrsMatrix>(
      const Teuchos::RCP<const Tpetra_CrsMatrix>& A,
      const std::string& prefix,
      int const counter)
  {
    if (A.is_null()) {
      return;
    }

    std::ostringstream oss;

    oss << prefix;
    if (counter >= 0) {
      oss << '-' << std::setfill('0') << std::setw(3) << counter;
    }
    oss << ".mm";

    const std::string& filename = oss.str();

    Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile(filename, A);
  }

  CmdLineArgs::CmdLineArgs(const std::string& default_xml_filename,
                                   const std::string& default_xml_filename2,
                                   const std::string& default_xml_filename3) :
    xml_filename(default_xml_filename),
    xml_filename2(default_xml_filename2),
    xml_filename3(default_xml_filename3),
    has_first_xml_file(false),
    has_second_xml_file(false),
    has_third_xml_file(false),
    vtune(false) {}

  void CmdLineArgs::parse_cmdline(int argc , char ** argv,
                                          std::ostream& os) {
    bool found_first_xml_file = false;
    bool found_second_xml_file = false;
    for (int arg=1; arg<argc; ++arg) {
      if(!std::strcmp(argv[arg],"--help")) {
        os << argv[0] << " [--vtune] [inputfile1.xml] [inputfile2.xml] [inputfile3.xml]\n";
        std::exit(1);
      }
      else if (!std::strcmp(argv[arg],"--vtune")) {
        vtune = true;
      }
      else {
        if (!found_first_xml_file) {
          xml_filename=argv[arg];
          found_first_xml_file = true;
          has_first_xml_file = true;
        }
        else if (!found_second_xml_file) {
          xml_filename2=argv[arg];
          found_second_xml_file = true;
          has_second_xml_file = true;
        }
        else {
          xml_filename3=argv[arg];
          has_third_xml_file = true;
        }
      }
    }
  }

  void connect_vtune(const int p_rank) {
    std::stringstream cmd;
    pid_t my_os_pid=getpid();
    const std::string vtune_loc = "amplxe-cl";
    const std::string output_dir = "./vtune/vtune.";
    cmd << vtune_loc
        << " -collect hotspots -result-dir " << output_dir << p_rank
        << " -target-pid " << my_os_pid << " &";
    if (p_rank == 0)
      std::cout << cmd.str() << std::endl;
    safe_system(cmd.str().c_str());
    safe_system("sleep 10");
  }

  void do_stack_trace() {

        void* callstack[128];
        int i, frames = backtrace(callstack, 128);
        char** strs = backtrace_symbols(callstack, frames);
        for (i = 0; i < frames; ++i) {
            printf("%s\n", strs[i]);
        }
        free(strs);
  }

void safe_fscanf(int nitems, FILE* file, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  int ret = vfscanf(file, format, ap);
  va_end(ap);
  ALBANY_ASSERT(ret == nitems,
      ret<<"=safe_fscanf("<<nitems<<", "<<file<<", \""<<format<<"\")");
}

void safe_sscanf(int nitems, const char* str, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  int ret = vsscanf(str, format, ap);
  va_end(ap);
  ALBANY_ASSERT(ret == nitems,
      ret<<"=safe_sscanf("<<nitems<<", \""<<str<<"\", \""<<format<<"\")");
}

void safe_fgets(char* str, int size, FILE* stream) {
  char* ret = fgets(str, size, stream);
  ALBANY_ASSERT(ret == str,
      ret<<"=safe_fgets("<<static_cast<void*>(str)<<", "<<size<<", "<<stream<<")");
}

void safe_system(char const* str) {
  ALBANY_ASSERT(str, "safe_system called with null command string\n");
  int ret = system(str);
  ALBANY_ASSERT(str, ret<<"=safe_system(\""<<str<<"\")");
}

void assert_fail(std::string const& msg) {
  std::cerr << msg;
  abort();
}

BuildType build_type(const BuildType value)
{
  static bool initialized_ = false;
  static BuildType value_ = value;
  if (!initialized_ && (value != BuildType::None)) {
    value_ = value;
    initialized_ = true;
  }
  return value_;
}

} // namespace Albany

