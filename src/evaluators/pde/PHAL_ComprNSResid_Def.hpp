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
ComprNSResid<EvalT, Traits>::
ComprNSResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  qFluct       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  qFluctGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  qFluctDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  force       (p.get<std::string>              ("Body Force Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  mu          (p.get<std::string>                   ("Viscosity Mu QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  lambda          (p.get<std::string>                   ("Viscosity Lambda QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  kappa          (p.get<std::string>                   ("Viscosity Kappa QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  tau11          (p.get<std::string>                   ("Stress Tau11 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  tau12       (p.get<std::string>                   ("Stress Tau12 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau13       (p.get<std::string>                   ("Stress Tau13 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau22       (p.get<std::string>                   ("Stress Tau22 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau23       (p.get<std::string>                   ("Stress Tau23 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau33       (p.get<std::string>                   ("Stress Tau33 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{

  //TO DOs: 
  //3D 

  this->addDependentField(qFluct.fieldTag());
  this->addDependentField(qFluctGrad.fieldTag());
  this->addDependentField(qFluctDot.fieldTag());
  this->addDependentField(force.fieldTag());
  this->addDependentField(mu.fieldTag());
  this->addDependentField(kappa.fieldTag());
  this->addDependentField(lambda.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());
  this->addDependentField(tau11.fieldTag());
  this->addDependentField(tau12.fieldTag());
  this->addDependentField(tau13.fieldTag());
  this->addDependentField(tau22.fieldTag());
  this->addDependentField(tau23.fieldTag());
  this->addDependentField(tau33.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("ComprNSResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");

  qFluct.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  gamma_gas = bf_list->get("Gamma", 1.4); 
  Rgas = bf_list->get("Gas constant R", 0.714285733);
  Pr = bf_list->get("Prandtl number Pr", 0.72); 
  Re = bf_list->get("Reynolds number Re", 1.0); 



std::cout << " vecDim = " << vecDim << std::endl;
std::cout << " numDims = " << numDims << std::endl;


if (vecDim != numDims+2) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::ComprNS constructor:  " <<
                                  "Invalid Parameter vecDim.  vecDim should be numDims + 2 = " << numDims + 2 << "." << std::endl);}  


}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qFluct,fm);
  this->utils.setFieldData(qFluctGrad,fm);
  this->utils.setFieldData(qFluctDot,fm);
  this->utils.setFieldData(force,fm);
  this->utils.setFieldData(mu,fm);
  this->utils.setFieldData(kappa,fm);
  this->utils.setFieldData(lambda,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(tau11,fm);
  this->utils.setFieldData(tau12,fm);
  this->utils.setFieldData(tau13,fm);
  this->utils.setFieldData(tau22,fm);
  this->utils.setFieldData(tau23,fm);
  this->utils.setFieldData(tau33,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  if (numDims == 2) { //2D case; order of variables is rho, u, v, T
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) = qFluctDot(cell,qp,0)*wBF(cell,node,qp);  //d(rho)/dt
             for (std::size_t i=1; i < vecDim; i++) {
                Residual(cell,node,i) = qFluct(cell,qp,0)*qFluctDot(cell,qp,i)*wBF(cell,node,qp); //rho*du_i/dt; rho*dT/dt 
             }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += qFluct(cell,qp,0)*(qFluctGrad(cell,qp,1,0)+qFluctGrad(cell,qp,2,1))*wBF(cell,node,qp) //rho*div(u)
                                      + (qFluct(cell,qp,1)*qFluctGrad(cell,qp,0,0) + qFluct(cell,qp,2)*qFluctGrad(cell,qp,0,1))*wBF(cell,node,qp) //u*d(rho)/dx + v*d(rho)/dy  
                                      + force(cell,qp,0)*wBF(cell,node,qp); //f0
             Residual(cell, node, 1) += qFluct(cell,qp,0)*(qFluct(cell,qp,1)*qFluctGrad(cell,qp,1,0) + qFluct(cell,qp,2)*qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) //rho*(u*du/dx + v*du/dy)
                                      + Rgas*(qFluct(cell,qp,0)*qFluctGrad(cell,qp,3,0) + qFluct(cell,qp,3)*qFluctGrad(cell,qp,0,0))*wBF(cell,node,qp) //R*(rho*dT/dx + T*d(rho)/dx) 
                                      + 1.0/Re*tau11(cell,qp)*wGradBF(cell,node,qp,0) //tau11
                                      + 1.0/Re*tau12(cell,qp)*wGradBF(cell,node,qp,1)//tau12
                                      + force(cell,qp,1)*wBF(cell,node,qp); //f1
             Residual(cell, node, 2) += qFluct(cell,qp,0)*(qFluct(cell,qp,1)*qFluctGrad(cell,qp,2,0) + qFluct(cell,qp,2)*qFluctGrad(cell,qp,2,1))*wBF(cell,node,qp) //rho*(u*dv/dx + v*dv/dy)
                                      + Rgas*(qFluct(cell,qp,0)*qFluctGrad(cell,qp,3,1) + qFluct(cell,qp,3)*qFluctGrad(cell,qp,0,1))*wBF(cell,node,qp) //R*(rho*dT/dy + T*d(rho)/dy)
                                      + 1.0/Re*tau12(cell,qp)*wGradBF(cell,node,qp,0) //tau21
                                      + 1.0/Re*tau22(cell,qp)*wGradBF(cell,node,qp,1) //tau22
                                      + force(cell,qp,2)*wBF(cell,node,qp); //f2
             Residual(cell, node, 3) += qFluct(cell,qp,0)*(qFluct(cell,qp,1)*qFluctGrad(cell,qp,3,0) + qFluct(cell,qp,2)*qFluctGrad(cell,qp,3,1) //rho*(u*dT/dx + v*dT/dy) 
                                      + (gamma_gas - 1.0)*qFluct(cell,qp,3)*(qFluctGrad(cell,qp,1,0) + qFluctGrad(cell,qp,2,1)))*wBF(cell,node,qp) //(gamma-1)*T*div(u)
                                      - (gamma_gas - 1.0)/Rgas*qFluctGrad(cell,qp,1,0)*1.0/Re*tau11(cell,qp)*wBF(cell,node,qp) //-(gamma-1)/R*du/dx*tau11 
                                      - (gamma_gas - 1.0)/Rgas*1.0/Re*qFluctGrad(cell,qp,2,0)*tau12(cell,qp)*wBF(cell,node,qp) //-(gamma-1)/R*(dv/dx*tau12)
                                      - (gamma_gas - 1.0)/Rgas*1.0/Re*qFluctGrad(cell,qp,1,1)*tau12(cell,qp)*wBF(cell,node,qp) //-(gamma-1)/R*(du/dy*tau12)
                                      - (gamma_gas - 1.0)/Rgas*qFluctGrad(cell,qp,2,1)*1.0/Re*tau22(cell,qp)*wBF(cell,node,qp) // -(gamma-1)/R*dv/dy*tau22
                                      + gamma_gas*kappa(cell,qp)/(Pr*Re)*(qFluctGrad(cell,qp,3,0)*wGradBF(cell,node,qp,0) + qFluctGrad(cell,qp,3,1)*wGradBF(cell,node,qp,1)) //gamma*kappa/(Pr*Re)*(Delta T)
                                      + force(cell,qp,3)*wBF(cell,node,qp);  //f3 
           } 
          } 
        }
     }
   else if (numDims == 3) { //3D case - TO IMPLEMENT
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             for (std::size_t i=0; i < vecDim; i++) {
                Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
             }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += 0.0;  
             Residual(cell, node, 1) += 0.0;  
             Residual(cell, node, 2) += 0.0;  
             Residual(cell, node, 3) += 0.0;  
             Residual(cell, node, 4) += 0.0;  
            } 
          } 
        }
     }
}

//**********************************************************************
}

