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
ReactDiffSystemResid<EvalT, Traits>::
ReactDiffSystemResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  U       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  UGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{



  this->addDependentField(U.fieldTag());
  this->addDependentField(UGrad.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("ReactDiffSystemResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  mu = bf_list->get("Viscosity mu", 0.1); 

  std::cout << " vecDim = " << vecDim << std::endl;
  std::cout << " numDims = " << numDims << std::endl;


}

//**********************************************************************
template<typename EvalT, typename Traits>
void ReactDiffSystemResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UGrad,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ReactDiffSystemResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i) = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //- mu*delta(u) = f
        double f = 1.0; 
        Residual(cell,node,0) += mu*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                 mu*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) +
                                 mu*UGrad(cell,qp,0,2)*wGradBF(cell,node,qp,2) -
                                 f*wBF(cell,node,qp); 
      }
    }        
  }
}

//**********************************************************************
}



