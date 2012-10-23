//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
GPAMResid<EvalT, Traits>::
GPAMResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  C          (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Cgrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  CDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(C);
  this->addDependentField(Cgrad);
  //this->addDependentField(CDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("GPAMResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  C.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void GPAMResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(C,fm);
  this->utils.setFieldData(Cgrad,fm);
  //this->utils.setFieldData(CDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void GPAMResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
              for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i)=0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<vecDim; i++) {
              for (std::size_t dim=0; dim<numDims; dim++) {
                Residual(cell,node,i) += Cgrad(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
    } } } } }

}

//**********************************************************************
}

