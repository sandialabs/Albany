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
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{

  //TO DOs: 
  //swap order of equations so continuity is first 
  //Viscosity evaluator (Sutherland's viscosity law)
  //Stress evaluator
  //3D 

  this->addDependentField(qFluct);
  this->addDependentField(qFluctGrad);
  this->addDependentField(qFluctDot);
  this->addDependentField(force);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("ComprNSResid"+PHX::TypeString<EvalT>::value);

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
  mu = bf_list->get("Viscocity mu", 0.0); 
  lambda = -2.0/3.0*mu; //Stokes' hypothesis
  kappa = bf_list->get("Diffusivity kappa", 0.0);  


cout << "mu: " << mu << endl; 
cout << "lambda: " << lambda << endl; 
cout << "kappa: " << kappa << endl; 


cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;


//if (vecDim != numDims+2) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
//                                  std::endl << "Error in PHAL::ComprNS constructor:  " <<
  //                                "Invalid Parameter vecDim.  vecDim should be numDims + 2 = " << numDims + 2 << "." << std::endl);}  


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
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  if (numDims == 2) { //2D case
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             for (std::size_t i=0; i < 2; i++) {
                Residual(cell,node,i) = qFluct(cell,qp,2)*qFluctDot(cell,qp,i)*wBF(cell,node,qp); //rho*du_i/dt 
             }
             Residual(cell,node,2) = qFluctDot(cell,qp,2)*wBF(cell,node,qp);  //d(rho)/dt
             Residual(cell,node,3) = qFluct(cell,qp,2)*qFluctDot(cell,qp,3)*wBF(cell,node,qp); //rho*dT/dt   
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += qFluct(cell,qp,2)*(qFluct(cell,qp,0)*qFluctGrad(cell,qp,0,0) + qFluct(cell,qp,1)*qFluctGrad(cell,qp,0,1))*wBF(cell,node,qp) //rho*(u*du/dx + v*du/dy)
                                      + Rgas*(qFluct(cell,qp,2)*qFluctGrad(cell,qp,3,0) + qFluct(cell,qp,3)*qFluctGrad(cell,qp,2,0))*wBF(cell,node,qp) //R*(rho*dT/dx + T*d(rho)/dx) 
                                      + (mu/Re*2.0*qFluctGrad(cell,qp,0,0) + lambda/Re*(qFluctGrad(cell,qp,0,0)+qFluctGrad(cell,qp,1,1)))*wGradBF(cell,node,qp,0) //tau11
                                      + mu/Re*(qFluctGrad(cell,qp,0,1)+qFluctGrad(cell,qp,1,0))*wGradBF(cell,node,qp,1)//tau12
                                      + force(cell,qp,0)*wBF(cell,node,qp); //f0
             Residual(cell, node, 1) += qFluct(cell,qp,2)*(qFluct(cell,qp,0)*qFluctGrad(cell,qp,1,0) + qFluct(cell,qp,1)*qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) //rho*(u*dv/dx + v*dv/dy)
                                      + Rgas*(qFluct(cell,qp,2)*qFluctGrad(cell,qp,3,1) + qFluct(cell,qp,3)*qFluctGrad(cell,qp,2,1))*wBF(cell,node,qp) //R*(rho*dT/dy + T*d(rho)/dy)
                                      + mu/Re*(qFluctGrad(cell,qp,0,1)+qFluctGrad(cell,qp,1,0))*wGradBF(cell,node,qp,0) //tau21
                                      + (mu/Re*2.0*qFluctGrad(cell,qp,1,1) + lambda/Re*(qFluctGrad(cell,qp,0,0)+qFluctGrad(cell,qp,1,1)))*wGradBF(cell,node,qp,1) //tau22
                                      + force(cell,qp,1)*wBF(cell,node,qp); //f1
             Residual(cell, node, 2) += qFluct(cell,qp,2)*(qFluctGrad(cell,qp,0,0)+qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) //rho*div(u)
                                      + (qFluct(cell,qp,0)*qFluctGrad(cell,qp,2,0) + qFluct(cell,qp,1)*qFluctGrad(cell,qp,2,1))*wBF(cell,node,qp) //u*d(rho)/dx + v*d(rho)/dy  
                                      + force(cell,qp,2)*wBF(cell,node,qp); //f2 
             Residual(cell, node, 3) += qFluct(cell,qp,2)*(qFluct(cell,qp,0)*qFluctGrad(cell,qp,3,0) + qFluct(cell,qp,1)*qFluctGrad(cell,qp,3,1) //rho*(u*dT/dx + v*dT/dy) 
                                      + (gamma_gas - 1.0)*qFluct(cell,qp,3)*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1)))*wBF(cell,node,qp) //(gamma-1)*T*div(u)
                                      - (gamma_gas - 1.0)/Rgas*qFluctGrad(cell,qp,0,0)*(mu/Re*2.0*qFluctGrad(cell,qp,0,0) + lambda/Re*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1)))*wBF(cell,node,qp) //-(gamma-1)/R*du/dx*tau11 
                                      - (gamma_gas - 1.0)/Rgas*mu/Re*qFluctGrad(cell,qp,1,0)*(qFluctGrad(cell,qp,0,1)+qFluctGrad(cell,qp,1,0))*wBF(cell,node,qp) //-(gamma-1)/R*(dv/dx*tau12)
                                      - (gamma_gas - 1.0)/Rgas*mu/Re*qFluctGrad(cell,qp,0,1)*(qFluctGrad(cell,qp,0,1)+qFluctGrad(cell,qp,1,0))*wBF(cell,node,qp) //-(gamma-1)/R*(du/dy*tau12)
                                      - (gamma_gas - 1.0)/Rgas*qFluctGrad(cell,qp,1,1)*(mu/Re*2.0*qFluctGrad(cell,qp,1,1) + lambda/Re*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1)))*wBF(cell,node,qp) // -(gamma-1)/R*dv/dy*tau22
                                      + gamma_gas*kappa/(Pr*Re)*(qFluctGrad(cell,qp,3,0)*wGradBF(cell,node,qp,0) + qFluctGrad(cell,qp,3,1)*wGradBF(cell,node,qp,1)) //gamma*kappa/(Pr*Re)*(Delta T)
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

