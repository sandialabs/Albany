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

#ifndef ALBANY_FULLSTATERECONSTRUCTOR_HPP
#define ALBANY_FULLSTATERECONSTRUCTOR_HPP

#include "NOX_Epetra_Observer.H"

#include "Epetra_Vector.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_Map;

namespace Albany {

class ReducedSpace;

class FullStateReconstructor : public NOX::Epetra::Observer {
public:
  FullStateReconstructor(const Teuchos::RCP<Teuchos::ParameterList> &params,
                         const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver,
                         const Epetra_Map &decoratedMap);

  //! Calls underlying observer then evaluates projection error
  virtual void observeSolution(const Epetra_Vector& solution);
  
  //! Calls underlying observer then evaluates projection error
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;
  Teuchos::RCP<NOX::Epetra::Observer> decoratedObserver_;

  Teuchos::RCP<ReducedSpace> reducedSpace_;
  
  static Teuchos::RCP<Teuchos::ParameterList> fillDefaultParams(const Teuchos::RCP<Teuchos::ParameterList> &params);
  static Teuchos::RCP<Epetra_MultiVector> createOrthonormalBasis(const Teuchos::RCP<Teuchos::ParameterList> &params,
                                                                 const Epetra_Map &map);

  Epetra_Vector lastFullSolution_;
  void computeLastFullSolution(const Epetra_Vector& reducedSolution);

  // Disallow copy & assignment
  FullStateReconstructor(const FullStateReconstructor &);
  FullStateReconstructor operator=(const FullStateReconstructor &);
};

} // end namespace Albany

#endif /* ALBANY_FULLSTATERECONSTRUCTOR_HPP */
