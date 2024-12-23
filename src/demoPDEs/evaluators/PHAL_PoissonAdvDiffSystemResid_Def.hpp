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
PoissonAdvDiffSystemResid<EvalT, Traits>::
PoissonAdvDiffSystemResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  Phi       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  PhiGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{



  this->addDependentField(Phi.fieldTag());
  this->addDependentField(PhiGrad.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("PoissonAdvDiffSystemResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  Phi.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  kappa = bf_list->get("Diffusivity", 4.4e9); 
  rhom = bf_list->get("Density of negative ions", 1e23); 
  
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << " vecDim = " << vecDim << "\n";
  *out << " numDims = " << numDims << "\n";
  

}

//**********************************************************************
template<typename EvalT, typename Traits>
void PoissonAdvDiffSystemResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Phi,fm);
  this->utils.setFieldData(PhiGrad,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PoissonAdvDiffSystemResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i) = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //- div(kappa*gradPhi) = rhop + rhom 
        Residual(cell,node,0) += kappa*PhiGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                 kappa*PhiGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) -
                                 rhom*wBF(cell,node,qp); 
      }
    }        
  }
}

//**********************************************************************
}



