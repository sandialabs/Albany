//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_FullStateReconstructor.hpp"

#include "Albany_ReducedSpace.hpp"

#include "Albany_BasisInputFile.hpp"
#include "Albany_MultiVectorOutputFile.hpp"
#include "Albany_MultiVectorOutputFileFactory.hpp"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;

FullStateReconstructor::FullStateReconstructor(const RCP<Teuchos::ParameterList> &params,
                                               const RCP<NOX::Epetra::Observer> &decoratedObserver,
                                               const Epetra_Map &decoratedMap) :
  params_(fillDefaultBasisInputParams(params)),
  decoratedObserver_(decoratedObserver),
  reducedSpace_(),
  lastFullSolution_(decoratedMap, false)
{
  const RCP<const Epetra_MultiVector> orthogonalBasis = readOrthonormalBasis(decoratedMap, params_);
  reducedSpace_ = rcp(new LinearReducedSpace(*orthogonalBasis));
}

void FullStateReconstructor::observeSolution(const Epetra_Vector& solution)
{
  computeLastFullSolution(solution);
  decoratedObserver_->observeSolution(lastFullSolution_);
}

void FullStateReconstructor::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  computeLastFullSolution(solution);
  decoratedObserver_->observeSolution(lastFullSolution_, time_or_param_val);
}

void FullStateReconstructor::computeLastFullSolution(const Epetra_Vector& reducedSolution)
{
  reducedSpace_->expansion(reducedSolution, lastFullSolution_);
}

} // end namespace Albany
