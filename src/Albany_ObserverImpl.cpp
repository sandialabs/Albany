//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ObserverImpl.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"


namespace Albany {

ObserverImpl::
ObserverImpl (const Teuchos::RCP<Application> &app)
  : StatelessObserverImpl(app)
{}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_Vector& nonOverlappedSolution,
                const Teuchos::Ptr<const Thyra_MultiVector>& nonOverlappedSolution_dxdp,
                const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDot,
                const Teuchos::Ptr<const Thyra_Vector>& nonOverlappedSolutionDotDot)
{
  app_->evaluateStateFieldManager (stamp,
                                   nonOverlappedSolution,
                                   nonOverlappedSolutionDot,
                                   nonOverlappedSolutionDotDot,
                                   nonOverlappedSolution_dxdp);

  app_->getStateMgr().updateStates();

  //! update distributed parameters in the mesh
  auto distParamLib = app_->getDistributedParameterLibrary();
  auto disc = app_->getDiscretization();
  distParamLib->scatter();
  for(auto it : *distParamLib) {
    disc->setField(*it.second->overlapped_vector(),
                    it.second->name(),
                   /*overlapped*/ true);
  }

  StatelessObserverImpl::observeSolution (stamp,
                                          nonOverlappedSolution,
                                          nonOverlappedSolution_dxdp,
                                          nonOverlappedSolutionDot,
                                          nonOverlappedSolutionDotDot);
}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_MultiVector& nonOverlappedSolution, 
                const Teuchos::Ptr<const Thyra_MultiVector>& nonOverlappedSolution_dxdp)
{
  app_->evaluateStateFieldManager(stamp, nonOverlappedSolution, 
                                  nonOverlappedSolution_dxdp);
  app_->getStateMgr().updateStates();
  StatelessObserverImpl::observeSolution(stamp, nonOverlappedSolution, 
                                         nonOverlappedSolution_dxdp);
}

void ObserverImpl::
parameterChanged(const std::string& param)
{
  //! If a parameter has changed in value, saved/unsaved fields must be updated
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << param << " has changed!" << std::endl;
  app_->getPhxSetup()->init_unsaved_param(param);
}

} // namespace Albany
