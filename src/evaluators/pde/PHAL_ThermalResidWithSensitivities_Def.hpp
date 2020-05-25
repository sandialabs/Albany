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
ThermalResidWithSensitivities<EvalT, Traits>::
ThermalResidWithSensitivities(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Source   (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  kappa(p.get<Teuchos::Array<double>>("Thermal Conductivity")),
  rho(p.get<double>("Density")),
  C(p.get<double>("Heat Capacity"))
{

  this->addDependentField(wBF);
  this->addDependentField(Tdot);
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
  else if (thermal_source == "1D Cost") {
    force_type = ONEDCOST;
    if (numDims > 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 1D Cost' is not valid for > 1 spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
  } 
  else if (thermal_source == "2D Cost Expt") {
    force_type = TWODCOSTEXPT;
    if (numDims > 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for 2 spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
  } 
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Thermal Source = " << thermal_source << "!  Valid options are: 'None', '1D Cost' and '2D Cost Expt'. \n"); 
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
  this->setName("ThermalResidWithSensitivities" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResidWithSensitivities<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(TResidual, fm);
}
// **********************************************************************
template<typename EvalT,typename Traits>
typename ThermalResidWithSensitivities<EvalT,Traits>::ScalarT&
ThermalResidWithSensitivities<EvalT,Traits>::getValue(const std::string &n)
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
void ThermalResidWithSensitivities<EvalT, Traits>::
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
  else if (force_type == ONEDCOST) { //Source term such that T = a*x*(1-x)*cos(2*pi*kappa[0]*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = -2.0*a*kappa[0]*(M_PI*x*(1-x)*sin(2.0*M_PI*kappa[0]*t/rho/C) - cos(2.0*M_PI*kappa[0]*t/rho/C)); 
      }
    }
  }
  else if (force_type == TWODCOSTEXPT) { //Source term such that T = a*x*(1-x)*y*(1-y)*
                                         //cos(2*pi*kappa[0]*t/rho/C)*exp(2*pi*kappa[1]*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const ScalarT y = coordVec(cell, qp, 1); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = 2.0*M_PI*a*x*(1.0-x)*y*(1.0-y)*exp(2.0*M_PI*kappa[1]*t/rho/C)*(kappa[1]*cos(2*M_PI*kappa[0]*t/rho/C)
                          -kappa[0]*sin(2.0*M_PI*kappa[0]*t/rho/C)) + 2.0*a*cos(2.0*M_PI*kappa[0]*t/rho/C)
                          *exp(2.0*M_PI*kappa[1]*t/rho/C)*(kappa[0]*y*(1-y) + kappa[1]*x*(1-x)); 
      }
    }
  }

  // Evaluate residual:
  // We are solving the following PDE
  // rho*CdT/dt - kappa_1*dT/dx - kappa_2*dT/dy - kappa_3*dT/dz = f in 3D
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      TResidual(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Time-derivative contribution to residual
        TResidual(cell, node) += rho * C * Tdot(cell, qp) * wBF(cell, node, qp) 
        // Source contribution to residual
                               - Source(cell,qp) * wBF(cell, node, qp); 
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

