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
