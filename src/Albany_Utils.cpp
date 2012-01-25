/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_Utils.hpp"
#include "Teuchos_TestForException.hpp"
#include <cstdlib>
#include <stdexcept>

  // Start of Utils to do with Communicators
#ifdef ALBANY_MPI

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

  Teuchos::RCP<Teuchos::Comm<int> > Albany::createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(mc)));
  }

#else

  const Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(const Epetra_Comm& ec) { return 1; }

  Albany_MPI_Comm Albany::getMpiCommFromEpetraComm(Epetra_Comm& ec) { return 1; }

  Teuchos::RCP<Epetra_Comm> Albany::createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc) {
    return Teuchos::rcp(new Epetra_SerialComm);
  }

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
      if(pos == string::npos)
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

