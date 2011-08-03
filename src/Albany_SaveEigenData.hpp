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
