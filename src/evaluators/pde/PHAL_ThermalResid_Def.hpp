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
ThermalResid<EvalT, Traits>::
ThermalResid(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
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
  rho(p.get<double>("Density")),
  C(p.get<double>("Heat Capacity")),
  disable_transient(p.get<bool>("Disable Transient")), 
  kappa_x(p.get<std::string>("Thermal Conductivity: kappa_x"), dl->shared_param)
{
  this->addDependentField(wBF);
  if (!disable_transient) this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(kappa_x);
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

  if (numDims > 1) {
    kappa_y = decltype(kappa_y)(p.get<std::string>("Thermal Conductivity: kappa_y"), dl->shared_param);
    this->addDependentField(kappa_y);
  }
  if (numDims > 2) {
    kappa_z = decltype(kappa_z)(p.get<std::string>("Thermal Conductivity: kappa_y"), dl->shared_param);
    this->addDependentField(kappa_z);
  }
  std::string thermal_source = p.get<std::string>("Thermal Source"); 
  if (thermal_source == "None") {
    force_type = NONE;
  }
  else if (thermal_source == "1D Cost") {
    force_type = ONEDCOST;
    if (disable_transient) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 1D Cost' is only valid for transient problems!\n"); 
    }
  } 
  else if (thermal_source == "2D Cost Expt") {
    force_type = TWODCOSTEXPT;
    if (numDims < 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for 2 or more spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
    if (disable_transient) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for transient problems!\n"); 
    }
  } 
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Thermal Source = " << thermal_source << "!  Valid options are: 'None', '1D Cost' and '2D Cost Expt'. \n"); 
  }

  this->setName("ThermalResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  if (!disable_transient) this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(TResidual, fm);
  this->utils.setFieldData(kappa_x, fm);
  if (numDims > 1) this->utils.setFieldData(kappa_y, fm);
  if (numDims > 2) this->utils.setFieldData(kappa_z, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
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
  else if (force_type == ONEDCOST) { //Source term such that T = a*x*(1-x)*cos(2*pi*kappa_x*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = -2.0*a*kappa_x(0)*(M_PI*x*(1-x)*sin(2.0*M_PI*kappa_x(0)*t/rho/C) - cos(2.0*M_PI*kappa_x(0)*t/rho/C)); 
      }
    }
  }
  else if (force_type == TWODCOSTEXPT) { //Source term such that T = a*x*(1-x)*y*(1-y)*
                                         //cos(2*pi*kappa_x*t/rho/C)*exp(2*pi*kappa_y*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const ScalarT y = coordVec(cell, qp, 1); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = 2.0*M_PI*a*x*(1.0-x)*y*(1.0-y)*exp(2.0*M_PI*kappa_y(0)*t/rho/C)*(kappa_y(0)*cos(2*M_PI*kappa_x(0)*t/rho/C)
                          -kappa_x(0)*sin(2.0*M_PI*kappa_x(0)*t/rho/C)) + 2.0*a*cos(2.0*M_PI*kappa_x(0)*t/rho/C)
                          *exp(2.0*M_PI*kappa_y(0)*t/rho/C)*(kappa_x(0)*y*(1-y) + kappa_y(0)*x*(1-x)); 
      }
    }
  }

  // Evaluate residual:
  // We are solving the following PDE
  // rho*CdT/dt - kappa_x*dT/dx - kappa_y*dT/dy - kappa_z*dT/dz = f in 3D

  if (!disable_transient) { //Inertia terms
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < numNodes; ++node) {
        TResidual(cell, node) = 0.0;
        for (std::size_t qp = 0; qp < numQPs; ++qp) {
          // Time-derivative contribution to residual
          TResidual(cell, node) += rho * C * Tdot(cell, qp) * wBF(cell, node, qp);
        }
      }
    }
  }
  //Diffusion and source terms 
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Source contribution to residual
        TResidual(cell, node) -= Source(cell,qp) * wBF(cell, node, qp); 
        // Diffusion part of residual
        TResidual(cell, node) += kappa_x(0) * TGrad(cell, qp, 0) * wGradBF(cell, node, qp, 0); 
        if (numDims > 1) {
          TResidual(cell, node) += kappa_y(0) * TGrad(cell, qp, 1) * wGradBF(cell, node, qp, 1); 
        }
        if (numDims > 2) {
          TResidual(cell, node) += kappa_z(0) * TGrad(cell, qp, 2) * wGradBF(cell, node, qp, 2); 
        }
      }
    }
  }

}

//**********************************************************************
}

