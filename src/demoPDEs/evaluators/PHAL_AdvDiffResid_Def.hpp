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
AdvDiffResid<EvalT, Traits>::
AdvDiffResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  U       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  UGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  UDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{



  this->addDependentField(U.fieldTag());
  this->addDependentField(UGrad.fieldTag());
  this->addDependentField(UDot.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("AdvDiffResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];


  a = bf_list->get("Advection a", 1.0); 
  b = bf_list->get("Advection b", 1.0); 
  mu = bf_list->get("Viscosity mu", 0.1); 
  useAugForm = bf_list->get("Use Augmented Form", false); 
  formType = bf_list->get("Augmented Form Type", 1);

  bool error = true; 
  if (formType == 1 || formType == 2) 
    error = false;  

  TEUCHOS_TEST_FOR_EXCEPTION(error, std::logic_error,
       "Invalid Augmented Form Type: " << formType << "; valid options are 1 and 2");
 

  std::cout << "a, b, mu: " << a << ", " << b << ", " << mu << std::endl; 
  std::cout << " vecDim = " << vecDim << std::endl;
  std::cout << " numDims = " << numDims << std::endl;
  std::cout << "Augmented Form? " << useAugForm << std::endl; 
  if (useAugForm == true) 
    std::cout << "  Augmented form type: " << formType << std::endl; 


}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UGrad,fm);
  this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  if (useAugForm == false) { //standard form of advection-diffusion equation
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t i=0; i<vecDim; i++)  
          Residual(cell,node,i) = 0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            //du/dt + a*du/dx + b*du/dy - mu*delta(u) = 0
            Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell,node,qp) + 
                                     a*UGrad(cell,qp,0,0)*wBF(cell,node,qp) +
                                     b*UGrad(cell,qp,0,1)*wBF(cell,node,qp) +  
                                     mu*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                     mu*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1); 
          }
       }        
    }
  }
  else { //augmented form of advection diffusion equation
    if (formType == 1) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++)  
            Residual(cell,node,i) = 0.0;
            for (std::size_t qp=0; qp < numQPs; ++qp) {
              //du/dt + (a,b).q - mu*div(q) = 0
              Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell,node,qp) + 
                                       a*U(cell,qp,1)*wBF(cell,node,qp) +
                                       b*U(cell,qp,2)*wBF(cell,node,qp) +  
                                       mu*U(cell,qp,1)*wGradBF(cell,node,qp,0) + 
                                       mu*U(cell,qp,2)*wGradBF(cell,node,qp,1);
              //q - grad(u) = 0 
              Residual(cell,node,1) += U(cell,qp,1)*wBF(cell,node,qp) - UGrad(cell,qp,0,0)*wBF(cell,node,qp); 
              Residual(cell,node,2) += U(cell,qp,2)*wBF(cell,node,qp) - UGrad(cell,qp,0,1)*wBF(cell,node,qp); 
            }
          }        
       }
    }
    else if (formType == 2) { 
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++)  
            Residual(cell,node,i) = 0.0;
            for (std::size_t qp=0; qp < numQPs; ++qp) {
              //du/dt + q = 0
              Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell,node,qp) + 
                                       U(cell,qp,1)*wBF(cell,node,qp) +
                                       U(cell,qp,2)*wBF(cell,node,qp);
              //q - (a,b).grad(u) + mu*delta(u) = 0
              Residual(cell,node,1) += U(cell,qp,1)*wBF(cell,node,qp) - a*UGrad(cell,qp,0,0)*wBF(cell,node,qp) 
                                    -  mu*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0); 
              Residual(cell,node,2) += U(cell,qp,2)*wBF(cell,node,qp) - b*UGrad(cell,qp,0,1)*wBF(cell,node,qp) 
                                    -  mu*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1);
            }
          }        
        }
      }
    }
}

//**********************************************************************
}



