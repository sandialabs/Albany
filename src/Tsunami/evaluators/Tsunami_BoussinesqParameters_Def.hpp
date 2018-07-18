//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"


namespace Tsunami {


template<typename EvalT, typename Traits>
BoussinesqParameters<EvalT, Traits>::
BoussinesqParameters(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  waterdepthQPin     (p.get<std::string>("Water Depth In QP Name"),dl->qp_scalar),
  zalphaQPin         (p.get<std::string>("z_alpha In QP Name"),dl->qp_scalar),
  waterdepthQP       (p.get<std::string>("Water Depth QP Name"),dl->qp_scalar),
  zalphaQP           (p.get<std::string>("z_alpha QP Name"),dl->qp_scalar),
  betaQP             (p.get<std::string>("Beta QP Name"),dl->qp_scalar),
  h                  (p.get<double>("Water Depth")), 
  zAlpha             (p.get<double>("Z_alpha")), 
  use_params_on_mesh (p.get<bool>("Use Parameters on Mesh")),
  enable_memoizer    (p.get<bool>("Enable Memoizer"))
{

  this->addDependentField(waterdepthQPin);
  this->addDependentField(zalphaQPin);

  this->addEvaluatedField(waterdepthQP);
  this->addEvaluatedField(zalphaQP);
  this->addEvaluatedField(betaQP);
  
  if (enable_memoizer)   
    memoizer.enable_memoizer();

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("BoussinesqParameters"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqParameters<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(waterdepthQPin,fm);
  this->utils.setFieldData(zalphaQPin,fm);
  this->utils.setFieldData(waterdepthQP,fm);
  this->utils.setFieldData(zalphaQP,fm);
  this->utils.setFieldData(betaQP,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqParameters<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //If memoizer is on, do this just once at the beginning of the simulation
  if (memoizer.have_stored_data(workset)) return;

  if (use_params_on_mesh == false) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        waterdepthQP(cell,qp) = ScalarT(h); 
        zalphaQP(cell,qp) = ScalarT(zAlpha); 
        betaQP(cell,qp) = ScalarT(zAlpha)/ScalarT(h); 
      }
    }
  }
  else {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        if (waterdepthQPin(cell,qp) <= 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
           "Invalid value of water depth field!  Water depth must be >0.\n");
        }
        if (zalphaQPin(cell,qp) > 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
           "Invalid value of z_alpha field!  z_alpha must be <=0.\n");
        }
        else {
          waterdepthQP(cell,qp) = waterdepthQPin(cell,qp); 
          zalphaQP(cell,qp) = zalphaQPin(cell,qp); 
          betaQP(cell,qp) = zalphaQPin(cell,qp)/waterdepthQPin(cell,qp); 
        }
      }
    }
  }
}

}
