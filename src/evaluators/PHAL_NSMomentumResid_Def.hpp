//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
NSMomentumResid<EvalT, Traits>::
NSMomentumResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ), 
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  pGrad       (p.get<std::string>                   ("Pressure Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  VGrad       (p.get<std::string>                   ("Velocity Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  P           (p.get<std::string>                   ("Pressure QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Rm          (p.get<std::string>              ("Rm Name"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  mu          (p.get<std::string>                   ("Viscosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  MResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ),
  haveSUPG(p.get<bool>("Have SUPG"))
{

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF.fieldTag());  
  this->addDependentField(pGrad.fieldTag());
  this->addDependentField(VGrad.fieldTag());
  this->addDependentField(wGradBF.fieldTag());
  this->addDependentField(P.fieldTag());
  this->addDependentField(Rm.fieldTag());
  this->addDependentField(mu.fieldTag());
  if (haveSUPG) {
    V = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(
      p.get<std::string>("Velocity QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    rho = PHX::MDField<ScalarT,Cell,QuadPoint>(
      p.get<std::string>("Density QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    TauM = PHX::MDField<ScalarT,Cell,QuadPoint>(
	p.get<std::string>("Tau M Name"),
	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(V.fieldTag());
    this->addDependentField(rho.fieldTag());
    this->addDependentField(TauM.fieldTag());
  }
 
  this->addEvaluatedField(MResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];
  
  this->setName("NSMomentumResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSMomentumResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(pGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(wGradBF,fm); 
  this->utils.setFieldData(P,fm);
  this->utils.setFieldData(Rm,fm);
  this->utils.setFieldData(mu,fm);
  if (haveSUPG) {
    this->utils.setFieldData(V,fm);
    this->utils.setFieldData(rho,fm);
    this->utils.setFieldData(TauM,fm);
  }
 
  this->utils.setFieldData(MResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSMomentumResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {          
      for (std::size_t i=0; i<numDims; i++) {
	MResidual(cell,node,i) = 0.0;
	for (std::size_t qp=0; qp < numQPs; ++qp) {
	  MResidual(cell,node,i) += 
	    (Rm(cell, qp, i)-pGrad(cell,qp,i))*wBF(cell,node,qp) -
	    P(cell,qp)*wGradBF(cell,node,qp,i);               
	  for (std::size_t j=0; j < numDims; ++j) { 
	    MResidual(cell,node,i) += 
	      mu(cell,qp)*(VGrad(cell,qp,i,j)+VGrad(cell,qp,j,i))*wGradBF(cell,node,qp,j);
//	      mu(cell,qp)*VGrad(cell,qp,i,j)*wGradBF(cell,node,qp,j);
	  }  
	}
      }
    }
  }
  
  if (haveSUPG) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
	for (std::size_t i=0; i<numDims; i++) {
	  for (std::size_t qp=0; qp < numQPs; ++qp) {           
	    for (std::size_t j=0; j < numDims; ++j) { 
	      MResidual(cell,node,i) += 
		rho(cell,qp)*TauM(cell,qp)*Rm(cell,qp,j)*V(cell,qp,j)*wGradBF(cell,node,qp,j);
	    }  
	  }
	}
      }
    }
  }
 
}

}

