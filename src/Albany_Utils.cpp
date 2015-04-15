//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Teuchos_TestForException.hpp"
#include <cstdlib>
#include <stdexcept>

  // Start of Utils to do with Communicators
#ifdef ALBANY_MPI

#if defined(ALBANY_EPETRA)
  const Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(const Epetra_Comm& ec) {
    const Epetra_MpiComm& emc = dynamic_cast<const Epetra_MpiComm&>(ec);
    return emc.Comm();
  }

  Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(Epetra_Comm& ec) {
    Epetra_MpiComm& emc = dynamic_cast<Epetra_MpiComm&>(ec);
    return emc.Comm();
  }

  Teuchos::RCP<Epetra_Comm> Albany::createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Epetra_MpiComm(mc));
  }

  Teuchos::RCP<Epetra_Comm> Albany::createEpetraCommFromTeuchosComm(const Teuchos::RCP<const Teuchos_Comm>& tc) {
    const Teuchos::Ptr<const Teuchos::MpiComm<int> > mpiComm =
               Teuchos::ptr_dynamic_cast<const Teuchos::MpiComm<int> >(Teuchos::ptrFromRef(*tc));
    return  Albany::createEpetraCommFromMpiComm(*mpiComm->getRawMpiComm()());
  }

  Teuchos::RCP<Teuchos_Comm> Albany::createTeuchosCommFromEpetraComm(const Teuchos::RCP<const Epetra_Comm>& ec) {
    const Teuchos::Ptr<const Epetra_MpiComm> mpiComm =
               Teuchos::ptr_dynamic_cast<const Epetra_MpiComm>(Teuchos::ptrFromRef(*ec));
    return  Albany::createTeuchosCommFromMpiComm(mpiComm->Comm());
  }

  Teuchos::RCP<Teuchos_Comm> Albany::createTeuchosCommFromEpetraComm(const Epetra_Comm& ec) {
    const Epetra_MpiComm *mpiComm =
               dynamic_cast<const Epetra_MpiComm *>(&ec);
    return  Albany::createTeuchosCommFromMpiComm(mpiComm->Comm());
  }
#endif


  Albany_MPI_Comm Albany::getMpiCommFromTeuchosComm(Teuchos::RCP<const Teuchos_Comm>& tc) {
    Teuchos::Ptr<const Teuchos::MpiComm<int> > mpiComm =
               Teuchos::ptr_dynamic_cast<const Teuchos::MpiComm<int> >(Teuchos::ptrFromRef(*tc));
    return *mpiComm->getRawMpiComm();

  }

  Teuchos::RCP<Teuchos::Comm<int> > Albany::createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(mc)));
  }

#else

#if defined(ALBANY_EPETRA)

  const Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(const Epetra_Comm& ec) { return 1; }

  Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(Epetra_Comm& ec) { return 1; }

  Teuchos::RCP<Epetra_Comm> Albany::createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Epetra_SerialComm);
  }

  Teuchos::RCP<Epetra_Comm> Albany::createEpetraCommFromTeuchosComm(const RCP<const Teuchos_Comm>& tc) {
    return Teuchos::rcp(new Epetra_SerialComm);
  }

  Teuchos::RCP<const Teuchos_Comm> Albany::createTeuchosCommFromEpetraComm(const RCP<const Epetra_Comm>& ec) {
    return Teuchos::rcp(new Teuchos::SerialComm<int>());
  }
#endif

  Teuchos::RCP<Teuchos::Comm<int> > Albany::createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Teuchos::SerialComm<int>());
  }

#endif

  // End of Utils to do with Communicators

  std::string Albany::strint(const std::string s, const int i) {
    std::ostringstream ss;
    ss << s << " " << i;
    return ss.str();
  }

  bool Albany::isValidInitString(const std::string& initString) {

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

  std::string Albany::doubleToInitString(double val) {
    std::string verbiage("initial value ");
    std::stringstream ss;
    ss << verbiage << val;
    return ss.str();
  }

  double Albany::initStringToDouble(const std::string& initString) {
    TEUCHOS_TEST_FOR_EXCEPTION(!Albany::isValidInitString(initString), std::range_error,
			       " initStringToDouble() called with invalid initialization string: " + initString + "\n");
    std::string verbiage("initial value ");
    std::string valueString = initString.substr(verbiage.size(), initString.size() - verbiage.size());
    return std::atof(valueString.c_str());
  }

  void Albany::splitStringOnDelim(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
  }

  void Albany::printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_Vector>& vec){

    Teuchos::ArrayRCP<const double> vv = vec->get1dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t i = 0; i < vec->getLocalLength(); i++){
       os.width(20);
       os << "             " << std::left << vv[i] << std::endl;
    }

  }

  void Albany::printTpetraVector(std::ostream &os, const Teuchos::Array<std::string>& names,
        const Teuchos::RCP<const Tpetra_Vector>& vec){

    Teuchos::ArrayRCP<const double> vv = vec->get1dView();

    os <<  std::setw(10) << std::endl;
    for(std::size_t i = 0; i < names.size(); i++){
       os.width(20);
//       os << "             " << std::left << vv[i] << std::endl;
       os << "   " << std::left << names[i] << "\t" << vv[i] << std::endl;
    }

  }

  void Albany::printTpetraVector(std::ostream &os, const Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > >& names,
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

  void Albany::printTpetraVector(std::ostream &os, const Teuchos::RCP<const Tpetra_MultiVector>& vec){

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
