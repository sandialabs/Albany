//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#define UNSTEADY

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//base flow values
const double ubar = 1.0; 
const double vbar = 1.0; 
const double wbar = 1.0; 
const double zetabar = 1.0; 
const double pbar = 0.0; //0.714285714285714;
//fluid parameters  
const double alpha = 1.0; 
const double gamma_gas = 1.4; //gas constant 


//**********************************************************************
template<typename EvalT, typename Traits>
LinComprNSResid<EvalT, Traits>::
LinComprNSResid(const Teuchos::ParameterList& p) :
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
  force       (p.get<std::string>              ("Body Force Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(C);
  this->addDependentField(Cgrad);
  this->addDependentField(CDot);
  this->addDependentField(force);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("LinComprNSResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  C.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;


if (numDims == 2 & vecDim != 3) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::LinComprNS constructor:  " <<
                                  "Invalid Parameter vecDim.  vecDim should be 3 for a 2D problem. " << std::endl);}  

if (numDims == 3) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::LinComprNS constructor:  " <<
                                  "3D case not implemented yet. " << std::endl);}  
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(C,fm);
  this->utils.setFieldData(Cgrad,fm);
  this->utils.setFieldData(CDot,fm);
  this->utils.setFieldData(force,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

   if (numDims == 2) { //2D case
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
#ifdef UNSTEADY
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             for (std::size_t i=0; i < numDims; ++i) {
                Residual(cell,node,i) = CDot(cell,qp,i)*wBF(cell,node,qp); 
             }
          }
#endif
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += ubar*Cgrad(cell,qp,0,0)*wBF(cell,node,qp) + vbar*Cgrad(cell,qp,0,1)*wBF(cell,node,qp) 
                                     + zetabar*Cgrad(cell,qp,2,0)*wBF(cell,node,qp) 
                                     + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + vbar*du'/dy + zetabar*dp'/dx + f0
             Residual(cell, node, 1) += ubar*Cgrad(cell,qp,1,0)*wBF(cell,node,qp) + vbar*Cgrad(cell,qp,1,1)*wBF(cell,node,qp) 
                                     + zetabar*Cgrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,1)*wBF(cell,node,qp); //ubar*dv'/dx + vbar*dv'/dy + zetabar*dp'/dy + f1
             Residual(cell, node, 2) += gamma_gas*pbar*(Cgrad(cell,qp,0,0) + Cgrad(cell,qp,1,1))*wBF(cell,node,qp) 
                                     + ubar*Cgrad(cell,qp,2,0)*wBF(cell,node,qp) + vbar*Cgrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,2)*wBF(cell,node,qp); //gamma*pbar*div(u') + ubar*dp'/dx + vbar*dp'/dy + f2
            } 
          } 
        }
    }
   else if (numDims == 3) { //3D case
   }
}

//**********************************************************************
}

