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
AdvectionResid<EvalT, Traits>::
AdvectionResid(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  udot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  uGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  C(p.get<double>("Heat Capacity")),
  rho(p.get<double>("Density")),
  conductivityIsDistParam(p.get<bool>("Distributed Thermal Conductivity")),
  disable_transient(p.get<bool>("Disable Transient")),
  source   (p.get<std::string>                   ("source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  residual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(wBF);
  if (!disable_transient) this->addDependentField(udot);
  this->addDependentField(uGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(source);
  this->addEvaluatedField(residual);
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

  if (!conductivityIsDistParam) {  
    kappa_x = decltype(kappa_x)(p.get<std::string>("Thermal Conductivity: kappa_x"), dl->shared_param);
    this->addDependentField(kappa_x);
    if (numDims > 1) {
      kappa_y = decltype(kappa_y)(p.get<std::string>("Thermal Conductivity: kappa_y"), dl->shared_param);
      this->addDependentField(kappa_y);
    }
    if (numDims > 2) {
      kappa_z = decltype(kappa_z)(p.get<std::string>("Thermal Conductivity: kappa_z"), dl->shared_param);
      this->addDependentField(kappa_z);
    }
  }
  else {  
    ThermalCond = decltype(ThermalCond)(p.get<std::string>("ThermalConductivity Name"),
  	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(ThermalCond);
  }

  std::string thermal_source = p.get<std::string>("Advection Source"); 
  if (thermal_source == "None") {
    force_type = NONE;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Advection Source = " << thermal_source << "!  Valid options are: 'None'. \n"); 
  }

  this->setName("AdvectionResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvectionResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(uGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  if (!disable_transient) this->utils.setFieldData(udot, fm);
  this->utils.setFieldData(source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(residual, fm);
  if (!conductivityIsDistParam) {
    this->utils.setFieldData(kappa_x, fm);
    if (numDims > 1) this->utils.setFieldData(kappa_y, fm);
    if (numDims > 2) this->utils.setFieldData(kappa_z, fm);
  }
  else {
    this->utils.setFieldData(ThermalCond, fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvectionResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Evaluate source term 
  if (force_type == NONE) { //No source term
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        source(cell, qp) = 0.0;
      }
    }
  }

  // Evaluate residual: for the following PDE
  // du/dt + a(x)*du/dx = 0

  if (!disable_transient) { //Inertia terms
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < numNodes; ++node) {
        residual(cell, node) = 0.0;
        for (std::size_t qp = 0; qp < numQPs; ++qp) {
          // Time-derivative contribution to residual
          residual(cell, node) += udot(cell, qp) * wBF(cell, node, qp);
        }
      }
    }
  }
  //Advection and source terms 
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // source contribution to residual
        residual(cell, node) -= source(cell,qp) * wBF(cell, node, qp); 
        // Diffusion part of residual
	residual(cell, node) += kappa_x(0) * uGrad(cell, qp, 0) * wBF(cell, node, qp); 
      }
    }
  }
}

//**********************************************************************
}

