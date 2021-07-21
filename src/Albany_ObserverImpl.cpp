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

void ObserverImpl::
parametersChanged()
{
  std::vector<std::string> p_names;

  const Teuchos::RCP<const Teuchos::ParameterList> pl = app_->getProblemPL();

  if (Teuchos::nonnull(pl)) {
    auto params_pl = pl->sublist("Parameters");
    if (params_pl.isParameter("Number Of Parameters"))
    {
      int num_parameters = params_pl.get<int>("Number Of Parameters");
      for (int i=0; i<num_parameters; ++i)
      {
        const Teuchos::ParameterList& pvi = params_pl.sublist(Albany::strint("Parameter",i));

        std::string parameterType = "Scalar";

        if(pvi.isParameter("Type"))
          parameterType = pvi.get<std::string>("Type");
        
        if (parameterType == "Scalar" || parameterType == "Distributed")
          p_names.push_back(pvi.get<std::string>("Name"));
        else {
          int m = pvi.get<int>("Dimension");
          for (int j=0; j<m; ++j)
          {
            const Teuchos::ParameterList& pj = pvi.sublist(Albany::strint("Scalar",j));
            p_names.push_back(pj.get<std::string>("Name"));
          }
        }
      }

      for (auto p_name : p_names)
        this->parameterChanged(p_name);
    }
  }
}

void ObserverImpl::
observeResponse(int iter)
{
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  for (int j = 0; j < app_->getNumResponses(); ++j) {
    *out << "Optimization Iteration " << iter << ", response " << j << ": ";
    app_->getResponse(j)->printResponse(out);
    *out << std::endl;
  }
}

} // namespace Albany
