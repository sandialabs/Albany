//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_SAVEEIGENDATA_HPP
#define ALBANY_SAVEEIGENDATA_HPP

#include "NOX_Common.H" // <string> and more
#include "NOX_Epetra_Observer.H"
#include "LOCA_SaveEigenData_AbstractStrategy.H" // base class
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_StateManager.hpp"

namespace Albany {

//! Strategy for saving eigenvector/value data
/*!
 * Saves eigenvectors and corresponding eigenvalues
 * using a custom strategy.
 */

class SaveEigenData : public LOCA::SaveEigenData::AbstractStrategy {

public:

  //! Constructor
  SaveEigenData(Teuchos::ParameterList& locaParams, 
		Teuchos::RCP<NOX::Epetra::Observer> observer = Teuchos::null,
		Albany::StateManager* pStateMgr = NULL);
    
  //! Destructor
  virtual ~SaveEigenData();

  //! Save eigenvalues/eigenvectors
  virtual NOX::Abstract::Group::ReturnType 
  save(Teuchos::RCP< std::vector<double> >& evals_r,
	 Teuchos::RCP< std::vector<double> >& evals_i,
	 Teuchos::RCP< NOX::Abstract::MultiVector >& evecs_r,
	 Teuchos::RCP< NOX::Abstract::MultiVector >& evecs_i);

private:

  //! Private to prohibit copying
  SaveEigenData(const SaveEigenData&);

  //! Private to prohibit copying
  SaveEigenData& operator = (const SaveEigenData&);

protected:

  //! number of eigenvalues/vectors to save
  int nsave;
  int nSaveAsStates;
  Teuchos::RCP<NOX::Epetra::Observer> noxObserver;
  Albany::StateManager* pAlbStateMgr;

}; // Class SaveEigenData
}
#endif
