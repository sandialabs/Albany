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

#include "Albany_ProjectionErrorObserver.hpp"

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

ProjectionErrorObserver::ProjectionErrorObserver(const RCP<ParameterList> &params,
                                                 const Teuchos::RCP<NOX::Epetra::Observer>& decoratedObserver,
                                                 const RCP<const Epetra_Map> &decoratedMap) :
  decoratedObserver_(decoratedObserver),
  projectionError_(params, decoratedMap)
{
   // Nothing to do
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution)
{
  decoratedObserver_->observeSolution(solution);
  projectionError_.process(solution);
}

void ProjectionErrorObserver::observeSolution(const Epetra_Vector& solution, double time_or_param_val)
{
  decoratedObserver_->observeSolution(solution, time_or_param_val);
  projectionError_.process(solution);
}

} // end namespace Albany
