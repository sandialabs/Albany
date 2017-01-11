//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UTILS_H
#define ALBANY_UTILS_H

// For cudaCheckError
#include <stdexcept>

#ifdef ALBANY_MPI
  #define Albany_MPI_Comm MPI_Comm
  #define Albany_MPI_COMM_WORLD MPI_COMM_WORLD
  #define Albany_MPI_COMM_NULL MPI_COMM_NULL
  //IKT, FIXME: remove || defined(ALBANY_ATO) below 
  #if defined(ALBANY_EPETRA) || defined(ALBANY_ATO) 
    #include "Epetra_MpiComm.h"
  #endif
  #include "Teuchos_DefaultMpiComm.hpp"
#else
  #define Albany_MPI_Comm int
  #define Albany_MPI_COMM_WORLD 0  // This is compatible with Dakota
  #define Albany_MPI_COMM_NULL 99
  //IKT, FIXME: remove || defined(ALBANY_ATO) below 
  #if defined(ALBANY_EPETRA) || defined(ALBANY_ATO) 
    #include "Epetra_SerialComm.h"
  #endif
  #include "Teuchos_DefaultSerialComm.hpp"
#endif
#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"

// Checks if the previous Kokkos::Cuda kernel has failed
#define cudaCheckError() { cudaError(__FILE__, __LINE__); }
inline void cudaError(const char *file, int line) {
#if defined(KOKKOS_HAVE_CUDA) && defined(ALBANY_CUDA_ERROR_CHECK)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s before %s:%d\n", cudaGetErrorString(err), file, line);
    throw std::runtime_error(cudaGetErrorString(err));
  }
#endif
}

namespace Albany {

  //Helper function which replaces the diagonal of a matrix 
  void
  ReplaceDiagonalEntries(const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
                         const Teuchos::RCP<Tpetra_Vector>& diag);

  //Helper function which computes absolute values of the rowsum
  //of a matrix, and puts it in a vector. 
  Teuchos::RCP<Tpetra_Vector> 
  InvRowSum(const Teuchos::RCP<const Tpetra_CrsMatrix>& matrix); 

//IKT, FIXME: ultimately get ride of || defined (ALBANY_ATO) below 
#if defined(ALBANY_EPETRA) || defined (ALBANY_ATO)

  Albany_MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec);

  Albany_MPI_Comm getMpiCommFromEpetraComm(Epetra_Comm& ec);
  Teuchos::RCP<Epetra_Comm> createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc);
  Teuchos::RCP<Epetra_Comm> createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc);
  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec);
  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromEpetraComm(const Epetra_Comm& ec);

#endif
  
  //Helper function which replaces the diagonal of a matrix 
  void ReplaceDiagonalEntries(const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
                              const Teuchos::RCP<Tpetra_Vector>& diag);

  //Helper function which creates diagonal vector with entries equal to the 
  //absolute value of the rowsum of a matrix.
  Teuchos::RCP<Tpetra_Vector> 
  InvRowSum(const Teuchos::RCP<const Tpetra_CrsMatrix>& matrix); 

  Albany_MPI_Comm getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc);

  Teuchos::RCP<Teuchos_Comm> createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc);

  //! Utility to make a string out of a string + int with a delimiter: strint("dog",2,' ') = "dog 2"
  //! The default delimiter is ' '. Potential delimiters include '_' - "dog_2"
  std::string strint(const std::string s, const int i, const char delim = ' ');

  //! Returns true of the given string is a valid initialization string of the format "initial value 1.54"
  bool isValidInitString(const std::string& initString);

  //! Converts a double to an initialization string:  doubleToInitString(1.54) = "initial value 1.54"
  std::string doubleToInitString(double val);

  //! Converts an init string to a double:  initStringToDouble("initial value 1.54") = 1.54
  double initStringToDouble(const std::string& initString);

  //! Splits a std::string on a delimiter
  void splitStringOnDelim(const std::string &s, char delim, std::vector<std::string> &elems);

  //! Nicely prints out a Tpetra Vector
  void printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_Vector>& vec);
  void printTpetraVector(std::ostream &os, const Teuchos::Array<std::string>& names,
         const Teuchos::RCP<const Tpetra_Vector>& vec);

  //! Nicely prints out a Tpetra MultiVector
  void printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_MultiVector>& vec);
  void printTpetraVector(std::ostream &os, const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > >& names,
         const Teuchos::RCP<const Tpetra_MultiVector>& vec);

  // Parses and stores command-line arguments
  struct CmdLineArgs {
    std::string xml_filename;
    std::string xml_filename2;
    std::string xml_filename3;
    bool has_first_xml_file;
    bool has_second_xml_file;
    bool has_third_xml_file;
    bool vtune;

    CmdLineArgs(const std::string& default_xml_filename = "input.xml",
                const std::string& default_xml_filename2 = "",
                const std::string& default_xml_filename3 = "");
    void parse_cmdline(int argc , char ** argv, std::ostream& os);
  };

  // Connect executable to vtune for profiling
  void connect_vtune(const int p_rank);

  // Do a nice stack trace for debugging
  void do_stack_trace();

  // Check returns codes and throw Teuchos exceptions
  // Useful for silencing compiler warnings about unused return codes
  void safe_fscanf(int nitems, FILE* file, const char* format, ...);
  void safe_sscanf(int nitems, const char* str, const char* format, ...);
  void safe_fgets(char* str, int size, FILE* stream);
  void safe_system(char const* str);
}
#endif //ALBANY_UTILS
