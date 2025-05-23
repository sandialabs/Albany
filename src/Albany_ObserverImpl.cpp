//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ObserverImpl.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StringUtils.hpp"

#include <stdexcept>


namespace Albany {

ObserverImpl::
ObserverImpl (const Teuchos::RCP<Application> &app)
  : StatelessObserverImpl(app)
{}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_Vector& x,
                const Teuchos::Ptr<const Thyra_MultiVector>& dxdp,
                const Teuchos::Ptr<const Thyra_Vector>& x_dot,
                const Teuchos::Ptr<const Thyra_Vector>& x_dotdot)
{
  app_->evaluateStateFieldManager (stamp, x, x_dot, x_dotdot, dxdp);

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
                                          Teuchos::rcpFromRef(x),
                                          Teuchos::rcpFromPtr(x_dot),
                                          Teuchos::rcpFromPtr(x_dotdot),
                                          Teuchos::rcpFromPtr(dxdp));
}

void ObserverImpl::
observeSolution(double stamp,
                const Thyra_MultiVector& x, 
                const Teuchos::Ptr<const Thyra_MultiVector>& dxdp)
{
  app_->evaluateStateFieldManager(stamp, x, dxdp);
  Teuchos::Ptr<const Thyra_Vector> x_dot,x_dotdot;

  int x_ncols = x.domain()->dim();
  if (x_ncols==1) {
    x_dot = x.col(1).ptr();
  } else if (x_ncols==2) {
    x_dot = x.col(1).ptr();
    x_dotdot = x.col(2).ptr();
  }
  observeSolution (stamp, *x.col(0), dxdp, x_dot, x_dotdot);
}

void ObserverImpl::
parameterChanged(const std::string& param)
{
  //! If a parameter has changed in value, saved/unsaved fields must be updated
  // TO DO: in the future, it would be great to only unsave parameters that have changed.
  if (app_->getAppPL()->sublist("Debug Output").get("Report Parameter Changes",true)) {
    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    if (app_->getParamLib()->isParameter(param)) {
      *out << "Scalar parameter '" << param << "' has changed! " << std::endl;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (! app_->getDistributedParameterLibrary()->has(param), std::runtime_error,
          "Error! Parameter '" + param + "' is not a scalar nor a distributed parameter. Please, contact developers.\n");
      *out << "Distributed parameter '" << param << "' has changed!\n";
    }
  }
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
        const Teuchos::ParameterList& pvi = params_pl.sublist(util::strint("Parameter",i));

        std::string parameterType = "Scalar";

        if(pvi.isParameter("Type"))
          parameterType = pvi.get<std::string>("Type");
        
        if (parameterType == "Scalar" || parameterType == "Distributed")
          p_names.push_back(pvi.get<std::string>("Name"));
        else {
          int m = pvi.get<int>("Dimension");
          for (int j=0; j<m; ++j)
          {
            const Teuchos::ParameterList& pj = pvi.sublist(util::strint("Scalar",j));
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
