//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace Tsunami {

//**********************************************************************
template<typename EvalT, typename Traits>
BoussinesqResid<EvalT, Traits>::
BoussinesqResid(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  EtaUE       (p.get<std::string>                   ("EtaUE QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  EtaUEDot       (p.get<std::string>                   ("EtaUE Dot QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  EtaUEGrad      (p.get<std::string>                   ("EtaUE Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  EtaUEDotGrad      (p.get<std::string>             ("EtaUE Dot Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  out                (Teuchos::VerboseObjectBase::getDefaultOStream()),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(EtaUE);
  this->addDependentField(EtaUEDot);
  this->addDependentField(EtaUEGrad);
  this->addDependentField(EtaUEDotGrad);

  this->addEvaluatedField(Residual);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  EtaUE.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  *out << "numNodes, numQPs, numDims, vecDim = " << numNodes << ", " 
       <<  numQPs << ", " << numDims << ", " << vecDim << "\n"; 

  this->setName("BoussinesqResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(EtaUE,fm);
  this->utils.setFieldData(EtaUEDot,fm);
  this->utils.setFieldData(EtaUEGrad,fm);
  this->utils.setFieldData(EtaUEDotGrad,fm);
  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Zero out residual
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int i=0; i<vecDim; i++) 
          Residual(cell,node,i) = 0.0; 

  //IKT, FIXME: fill in!
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int i=0; i<vecDim; i++) {
        for (int qp=0; qp < numQPs; ++qp) {
          Residual(cell,node,i) += EtaUEDot(cell,qp,i)*wBF(cell,node,qp);
        }
      }
    }
  }
}

}

