//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_COUPLEDPSOBSERVER_HPP
#define QCAD_COUPLEDPSOBSERVER_HPP

#include "Epetra_Vector.h"
#include "NOX_Epetra_Observer.H"
#include "Piro_ProviderBase.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "QCAD_CoupledPoissonSchrodinger.hpp"

namespace QCAD {


  //! Coupled Poisson-Schrodinger observer
class CoupledPS_NOXObserver : public NOX::Epetra::Observer
{
public:
   CoupledPS_NOXObserver (const Teuchos::RCP<CoupledPoissonSchrodinger> &psModel);
   ~CoupledPS_NOXObserver () { };

  //! Original version, for steady with no time or param info
  void observeSolution( const Epetra_Vector& solution);

  //! Improved version with space for time or parameter value
  void observeSolution( const Epetra_Vector& solution, double time_or_param_val);

private:
  Teuchos::RCP<CoupledPoissonSchrodinger> psModel_;
};




  //! Coupled Poisson-Schrodinger observer factory (creates observer)
class CoupledPS_NOXObserverFactory {
public:
  explicit CoupledPS_NOXObserverFactory(const Teuchos::RCP<CoupledPoissonSchrodinger> &psModel);
  Teuchos::RCP<NOX::Epetra::Observer> createInstance();

private:
  Teuchos::RCP<CoupledPoissonSchrodinger> psModel_;
};




  //! Coupled Poisson-Schrodinger observer constructor (wrapper around factory)
class CoupledPS_NOXObserverConstructor : public Piro::ProviderBase<NOX::Epetra::Observer> {
public:
  explicit CoupledPS_NOXObserverConstructor(const Teuchos::RCP<CoupledPoissonSchrodinger> &psModel) :
    factory_(psModel),
    instance_(Teuchos::null)
  {}

  virtual Teuchos::RCP<NOX::Epetra::Observer> getInstance(
      const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  CoupledPS_NOXObserverFactory factory_;
  Teuchos::RCP<NOX::Epetra::Observer> instance_;
};



} // namespace QCAD

#endif

