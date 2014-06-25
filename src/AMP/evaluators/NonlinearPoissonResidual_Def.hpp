//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
NonlinearPoissonResidual<EvalT, Traits>::
NonlinearPoissonResidual(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  w_bf_       (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  w_grad_bf_  (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
  u_          (p.get<std::string>("Unknown Name"), dl->qp_scalar),
  u_grad_     (p.get<std::string>("Unknown Gradient Name"), dl->qp_vector),
  Tdot        (p.get<std::string>("QP Time Derivative Variable Name"), dl->qp_scalar),
  ThermalCond (p.get<std::string>("Thermal Conductivity Name"), dl->qp_scalar),
  Source      (p.get<std::string>("Source Name"), dl->qp_scalar),
  TResidual   (p.get<std::string>("Residual Name"), dl->node_scalar),
  haveSource  (p.get<bool>("Have Source")),
  haveConvection(false),
  haveAbsorption  (p.get<bool>("Have Absorption")),
  haverhoCp(false)
{

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(w_bf_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(u_);
  this->addDependentField(u_grad_);
  
  
  this->addDependentField(ThermalCond);
  if (enableTransient) this->addDependentField(Tdot);
  if (haveSource) this->addDependentField(Source);
  if (haveAbsorption) {
    Absorption = PHX::MDField<ScalarT,Cell,QuadPoint>(
	p.get<std::string>("Absorption Name"),
  dl->qp_scalar);
    this->addDependentField(Absorption);
  }
  this->addEvaluatedField(TResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];


  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);

  if (haveAbsorption)  aterm.resize(dims[0], numQPs);

  convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
                           "Convection Velocity", numDims, false);
  if (p.isType<std::string>("Convection Velocity")) {
    convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
                             "Convection Velocity", numDims, false);
  }
  if (convectionVels.size()>0) {
    haveConvection = true;
    if (p.isType<bool>("Have Rho Cp"))
      haverhoCp = p.get<bool>("Have Rho Cp");
    if (haverhoCp) {
      PHX::MDField<ScalarT,Cell,QuadPoint> tmp(p.get<std::string>("Rho Cp Name"),
            dl->qp_scalar);
      rhoCp = tmp;
      this->addDependentField(rhoCp);
    }
  }

  this->setName("NonlinearPoissonResidual"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_,fm);
  this->utils.setFieldData(w_grad_bf_,fm);
  this->utils.setFieldData(u_,fm);
  this->utils.setFieldData(u_grad_,fm);
  this->utils.setFieldData(ThermalCond,fm);
  
  
  
  if (haveSource)  this->utils.setFieldData(Source,fm);
  if (enableTransient) this->utils.setFieldData(Tdot,fm);

  if (haveAbsorption)  this->utils.setFieldData(Absorption,fm);

  if (haveConvection && haverhoCp)  this->utils.setFieldData(rhoCp,fm);

  this->utils.setFieldData(TResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // density residual
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      TResidual(cell,node) = 0.0;
    }
    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      for (std::size_t node = 0; node < numNodes; ++node) {
        for (std::size_t i = 0; i < numDims; ++i) {
          TResidual(cell,node) +=
            (1.0 + u_(cell,qp)*u_(cell,qp)) *
            u_grad_(cell,qp,i) * w_grad_bf_(cell,node,qp,i);
        }
      }
    }
  }

}

//**********************************************************************
}

