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
NavierStokesParameters<EvalT, Traits>::
NavierStokesParameters(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  viscosityQPin      (p.get<std::string>("Fluid Viscosity In QP Name"),dl->qp_scalar),
  densityQPin        (p.get<std::string>("Fluid Density In QP Name"),dl->qp_scalar),
  viscosityQP        (p.get<std::string>("Fluid Viscosity QP Name"),dl->qp_scalar),
  densityQP          (p.get<std::string>("Fluid Density QP Name"),dl->qp_scalar),
  mu                 (p.get<double>("Viscosity")), 
  rho                (p.get<double>("Density")), 
  use_params_on_mesh (p.get<bool>("Use Parameters on Mesh")),
  enable_memoizer    (p.get<bool>("Enable Memoizer"))
{

  this->addDependentField(viscosityQPin);
  this->addDependentField(densityQPin);
  this->addEvaluatedField(viscosityQP);
  this->addEvaluatedField(densityQP);
  
  if (enable_memoizer)   
    memoizer.enable_memoizer();

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("NavierStokesParameters"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesParameters<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(viscosityQPin,fm);
  this->utils.setFieldData(densityQPin,fm);
  this->utils.setFieldData(viscosityQP,fm);
  this->utils.setFieldData(densityQP,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesParameters<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //If memoizer is on, do this just once at the beginning of the simulation
  if (memoizer.have_stored_data(workset)) return;

  if (use_params_on_mesh == false) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        densityQP(cell,qp) = ScalarT(rho); 
        viscosityQP(cell,qp) = ScalarT(mu); 
      }
    }
  }
  else {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        if (densityQPin(cell,qp) <= 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
           "Invalid value of density field!  Density must be >0.\n");
        }
        else densityQP(cell,qp) = densityQPin(cell,qp); 
        if (viscosityQPin(cell,qp) <= 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
           "Invalid value of viscosity field!  Viscosity must be >0.\n");
        }
        else viscosityQP(cell,qp) = viscosityQPin(cell,qp); 
      }
    }
  }
}

}
