//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
SteadyThermalResidWithSensitivities<EvalT, Traits>::
SteadyThermalResidWithSensitivities(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Source   (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  kappa(p.get<Teuchos::Array<double>>("Thermal Conductivity"))
{

  this->addDependentField(wBF);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(Source);
  this->addEvaluatedField(TResidual);
  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  coordVec = decltype(coordVec)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  this->addDependentField(coordVec);

  Teuchos::RCP<PHX::DataLayout> node_vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  node_vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];

  if (kappa.size() != numDims) {      
    TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Thermal Conductivity size " << kappa.size() << " != # dimensions " << numDims << "\n"); 
  }
  
  std::string thermal_source = p.get<std::string>("Thermal Source"); 
  if (thermal_source == "None") {
    force_type = NONE;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Thermal Source = " << thermal_source << "!  Valid options are: 'None'\n"); 
  }

  // Add kappa[0] wavelength as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library");
  this->registerSacadoParameter("kappa_x", paramLib);
  if (numDims > 1) {
    // Add kappa[1] wavelength as a Sacado-ized parameter
    this->registerSacadoParameter("kappa_y", paramLib);
  }
  if (numDims > 2) {
    // Add kappa[2] wavelength as a Sacado-ized parameter
    this->registerSacadoParameter("kappa_z", paramLib);
  }
  this->setName("SteadyThermalResidWithSensitivities" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SteadyThermalResidWithSensitivities<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(TResidual, fm);
}
// **********************************************************************
template<typename EvalT,typename Traits>
typename SteadyThermalResidWithSensitivities<EvalT,Traits>::ScalarT&
SteadyThermalResidWithSensitivities<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "kappa_x") {
    value = kappa[0]; 
  }
  else if (n == "kappa_y") {
    value = kappa[1]; 
  }
  else if (n == "kappa_z") {
    value = kappa[2]; 
  }
  return value;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void SteadyThermalResidWithSensitivities<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Evaluate source term 
  if (force_type == NONE) { //No source term
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        Source(cell, qp) = 0.0;
      }
    }
  }

  // Evaluate residual:
  // We are solving the following PDE
  // - kappa_1*dT/dx - kappa_2*dT/dy - kappa_3*dT/dz = f in 3D
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      TResidual(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Source contribution to residual
        TResidual(cell, node) -= Source(cell,qp) * wBF(cell, node, qp); 
        // Diffusion part of residual
        for (std::size_t ndim = 0; ndim < numDims; ++ndim) {
          TResidual(cell, node) += kappa[ndim] * TGrad(cell, qp, ndim) *
                                   wGradBF(cell, node, qp, ndim);
        }
      }
    }
  }

}

//**********************************************************************
}

