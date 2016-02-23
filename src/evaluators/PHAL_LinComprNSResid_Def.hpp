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
LinComprNSResid<EvalT, Traits>::
LinComprNSResid(const Teuchos::ParameterList& p) :
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


  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(qFluct);
  this->addDependentField(qFluctGrad);
  if(enableTransient)
    this->addDependentField(qFluctDot);
  this->addDependentField(force);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("LinComprNSResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");
  std::string eqnType = bf_list->get("Type", "Euler");
  
  if (eqnType == "Euler") {
    std::cout << "setting euler equations!" << std::endl; 
    eqn_type = EULER; 
  }
  else if (eqnType == "Navier-Stokes") {
    std::cout << "setting n-s equations!" << std::endl; 
    eqn_type = NS; 
  }


  qFluct.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  Teuchos::Array<double> defaultBaseFlowData(numDims+2);  
  baseFlowData = bf_list->get("Base Flow Data", defaultBaseFlowData); 
  //for EULER, baseFlowData = (ubar, vbar, wbar, zetabar, pbar)
  //for NS, baseFlowData = (ubar, vbar, wbar, Tbar, rhobar)

  gamma_gas = bf_list->get("Gamma", 1.4); 
  Rgas = bf_list->get("Gas constant R", 0.714285733);
  Pr = bf_list->get("Prandtl number Pr", 0.72); 
  Re = bf_list->get("Reynolds number Re", 1.0); 
  mu = bf_list->get("Viscocity mu", 0.0); 
  lambda = -2.0/3.0*mu; //Stokes' hypothesis
  kappa = bf_list->get("Diffusivity kappa", 0.0);  
  IBP_convect_terms = bf_list->get("IBP Convective Terms", false); 

  if (IBP_convect_terms == true)
    std::cout  << "Integrating convective terms by parts in weak form." << std::endl; 


std::cout << " vecDim = " << vecDim << std::endl;
std::cout << " numDims = " << numDims << std::endl;

if (baseFlowData.size()!=numDims+2) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::LinComprNS constructor:  " <<
                                  "baseFlow data should have length numDims + 2 =  " << numDims+2 << "." << std::endl);} 


if (eqn_type == EULER & vecDim != numDims+1) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::LinComprNS constructor:  " <<
                                  "Invalid Parameter vecDim.  vecDim should be numDims + 1 = " << numDims + 1 << " for Euler equations." << std::endl);}  

if (eqn_type == NS & vecDim != numDims+2) {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in PHAL::LinComprNS constructor:  " <<
                                  "Invalid Parameter vecDim.  vecDim should be numDims + 2 = " << numDims + 2 << " for Navier-Stokes equations." << std::endl);}  

}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qFluct,fm);
  this->utils.setFieldData(qFluctGrad,fm);
  if(enableTransient)
    this->utils.setFieldData(qFluctDot,fm);
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
  typedef Intrepid2::FunctionSpaceTools FST;

  if (eqn_type == EULER) { //Euler equations
   if (numDims == 1) { //1D case
    double ubar = baseFlowData[0];
    double zetabar = baseFlowData[1]; 
    double pbar = baseFlowData[2];
    if (IBP_convect_terms == false) {//variational formulation in which the convective terms are not integrated by parts
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += ubar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp)  
                                     + zetabar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp)  
                                     + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + zetabar*dp'/dx + f0
             Residual(cell, node, 1) += gamma_gas*pbar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp) 
                                     + ubar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp)  
                                     + force(cell,qp,1)*wBF(cell,node,qp); //gamma*pbar*du'/dx + ubar*dp'/dx + f2
             
            } 
          } 
        }
     }
     else { //variational formulation in which the convective terms are integrated by parts
       for (std::size_t cell=0; cell < workset.numCells; ++cell) {
         for (std::size_t node=0; node < numNodes; ++node) {
           for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
           if(enableTransient)
             for (std::size_t qp=0; qp < numQPs; ++qp) {
                for (std::size_t i=0; i < vecDim; i++) {
                   Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
                }
           }
           for (std::size_t qp=0; qp < numQPs; ++qp) {
              Residual(cell, node, 0) += -1.0*ubar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,0)  
                                      - zetabar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,0) 
                                      + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + zetabar*dp'/dx + f0
              Residual(cell, node, 1) += -1.0*gamma_gas*pbar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,0)  
                                      - ubar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,0)  
                                      + force(cell,qp,1)*wBF(cell,node,qp); //gamma*pbar*du'/dx + ubar*dp'/dx  + f2
            } 
          } 
        }
     }
    }
   if (numDims == 2) { //2D case
    double ubar = baseFlowData[0]; 
    double vbar = baseFlowData[1]; 
    double zetabar = baseFlowData[2]; 
    double pbar = baseFlowData[3];
    if (IBP_convect_terms == false) {//variational formulation in which the convective terms are not integrated by parts
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += ubar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,0,1)*wBF(cell,node,qp) 
                                     + zetabar*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) 
                                     + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + vbar*du'/dy + zetabar*dp'/dx + f0
             Residual(cell, node, 1) += ubar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,1,1)*wBF(cell,node,qp) 
                                     + zetabar*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,1)*wBF(cell,node,qp); //ubar*dv'/dx + vbar*dv'/dy + zetabar*dp'/dy + f1
             Residual(cell, node, 2) += gamma_gas*pbar*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) 
                                     + ubar*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,2)*wBF(cell,node,qp); //gamma*pbar*div(u') + ubar*dp'/dx + vbar*dp'/dy + f2
            } 
          } 
        }
     }
     else { //variational formulation in which the convective terms are integrated by parts
       for (std::size_t cell=0; cell < workset.numCells; ++cell) {
         for (std::size_t node=0; node < numNodes; ++node) {
           for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
           if(enableTransient)
             for (std::size_t qp=0; qp < numQPs; ++qp) {
                for (std::size_t i=0; i < vecDim; i++) {
                   Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
                }
           }
           for (std::size_t qp=0; qp < numQPs; ++qp) {
              Residual(cell, node, 0) += -1.0*ubar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,1) 
                                      - zetabar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,0) 
                                      + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + vbar*du'/dy + zetabar*dp'/dx + f0
              Residual(cell, node, 1) += -1.0*ubar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,1) 
                                      - zetabar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,1) 
                                      + force(cell,qp,1)*wBF(cell,node,qp); //ubar*dv'/dx + vbar*dv'/dy + zetabar*dp'/dy + f1
              Residual(cell, node, 2) += -1.0*gamma_gas*pbar*(qFluct(cell,qp,0)*wGradBF(cell,node,qp,0) + qFluct(cell,qp,1)*wGradBF(cell,node,qp,1)) 
                                      - ubar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,1) 
                                      + force(cell,qp,2)*wBF(cell,node,qp); //gamma*pbar*div(u') + ubar*dp'/dx + vbar*dp'/dy + f2
            } 
          } 
        }
     }
    }
   else if (numDims == 3) { //3D case
    double ubar = baseFlowData[0]; 
    double vbar = baseFlowData[1]; 
    double wbar = baseFlowData[2]; 
    double zetabar = baseFlowData[3]; 
    double pbar = baseFlowData[4];
    if (IBP_convect_terms == false) {//variational formulation in which the convective terms are not integrated by parts
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += ubar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,0,1)*wBF(cell,node,qp) 
                                     +wbar*qFluctGrad(cell,qp,0,2)*wBF(cell,node,qp) + zetabar*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) 
                                     + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + vbar*du'/dy + wbar*du'/dz + zetabar*dp'/dx + f0
             Residual(cell, node, 1) += ubar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,1,1)*wBF(cell,node,qp) 
                                     + wbar*qFluctGrad(cell,qp,1,2)*wBF(cell,node,qp) + zetabar*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,1)*wBF(cell,node,qp); //ubar*dv'/dx + vbar*dv'/dy + wbar*dv'/dz + zetabar*dp'/dy + f1
             Residual(cell, node, 2) += ubar*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp)
                                     + wbar*qFluctGrad(cell,qp,2,2)*wBF(cell,node,qp) + zetabar*qFluctGrad(cell,qp,3,2)*wBF(cell,node,qp)
                                     + force(cell,qp,2)*wBF(cell,node,qp); //ubar*dw'/dx + vbar*dw'/dy + wbar*dw'/dz + zetabar*dp'/dz + f2
             Residual(cell, node, 3) += gamma_gas*pbar*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1) + qFluctGrad(cell,qp,2,2))*wBF(cell,node,qp) 
                                     + ubar*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) + wbar*qFluctGrad(cell,qp,3,2)*wBF(cell,node,qp) 
                                     + force(cell,qp,3)*wBF(cell,node,qp); //gamma*pbar*div(u') + ubar*dp'/dx + vbar*dp'/dy + wbar*dp'/dz + f3
            } 
          } 
        }
     }
     else { //variational formulation in which the convective terms are integrated by parts
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += -1.0*ubar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,1) 
                                     - wbar*qFluct(cell,qp,0)*wGradBF(cell,node,qp,2) - zetabar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,0) 
                                     + force(cell,qp,0)*wBF(cell,node,qp); //ubar*du'/dx + vbar*du'/dy + wbar*du'/dz + zetabar*dp'/dx + f0
             Residual(cell, node, 1) += -1.0*ubar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,1) 
                                     - wbar*qFluct(cell,qp,1)*wGradBF(cell,node,qp,2) - zetabar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,1) 
                                     + force(cell,qp,1)*wBF(cell,node,qp); //ubar*dv'/dx + vbar*dv'/dy + wbar*dv'/dz + zetabar*dp'/dy + f1
             Residual(cell, node, 2) += -1.0*ubar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,1)
                                     - wbar*qFluct(cell,qp,2)*wGradBF(cell,node,qp,2) - zetabar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,2)
                                     + force(cell,qp,2)*wBF(cell,node,qp); //ubar*dw'/dx + vbar*dw'/dy + wbar*dw'/dz + zetabar*dp'/dz + f2
             Residual(cell, node, 3) += -1.0*gamma_gas*pbar*(qFluct(cell,qp,0)*wGradBF(cell,node,qp,0) + qFluct(cell,qp,1)*wGradBF(cell,node,qp,1) + qFluct(cell,qp,2)*wGradBF(cell,node,qp,2)) 
                                     - ubar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,0) - vbar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,1) - wbar*qFluct(cell,qp,3)*wGradBF(cell,node,qp,2) 
                                     + force(cell,qp,3)*wBF(cell,node,qp); //gamma*pbar*div(u') + ubar*dp'/dx + vbar*dp'/dy + wbar*dp'/dz + f3
            } 
          } 
        }
     }
   }
  }
  else if (eqn_type == NS) { //Navier-Stokes equations
    if (numDims == 2) { //2D case
      double ubar = baseFlowData[0]; 
      double vbar = baseFlowData[1]; 
      double Tbar = baseFlowData[2]; 
      double rhobar = baseFlowData[3];
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = rhobar*qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
                  if (i == vecDim-1)
                    Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += rhobar*ubar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,0,1)*wBF(cell,node,qp) 
                                     + rhobar*Rgas*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) + Rgas*Tbar*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) 
                                     + 1.0/Re*((2.0*mu + lambda)*qFluctGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + lambda*qFluctGrad(cell,qp,1,1)*wGradBF(cell,node,qp,0) 
                                               + mu*qFluctGrad(cell,qp,1,0)*wGradBF(cell,node,qp,1) + mu*qFluctGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1))
                                     + force(cell,qp,0)*wBF(cell,node,qp); //rhobar*ubar*du'/dx + rhobar*vbar*du'/dy + rhobar*R*dT'/dx + R*Tbar*drho'/dx + 
                                                                           //1/Re*(d/dx((2*mu+lambda)*du'/dx + lambda*dv'/dy) + d/dy*(mu*dv'/dx + mu*du'/dy)) + 
                                                                           //f0
             Residual(cell, node, 1) += rhobar*ubar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,1,1)*wBF(cell,node,qp) 
                                     + rhobar*Rgas*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp) + Rgas*Tbar*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) 
                                     + 1.0/Re*(mu*qFluctGrad(cell,qp,1,0)*wGradBF(cell,node,qp,0) + mu*qFluctGrad(cell,qp,0,1)*wGradBF(cell,node,qp,0)
                                               + lambda*qFluctGrad(cell,qp,0,0)*wGradBF(cell,node,qp,1) + (2.0*mu + lambda)*qFluctGrad(cell,qp,1,1)*wGradBF(cell,node,qp,1))
                                     + force(cell,qp,1)*wBF(cell,node,qp); //rhobar*ubar*dv'/dx + rhobar*vbar*dv'/dy + rhobar*R*dT'/dy + R*Tbar*drho'/dy + 
                                                                           //1/Re*(d/dx(mu*dv'/dx + mu*du'/dy) + d/dy(lambda*du'/dx + (2*mu + lambda)*dv'/dy)) +
                                                                           //f1
             Residual(cell, node, 2) += rhobar*Tbar*(gamma_gas - 1.0)*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) 
                                     + rhobar*ubar*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + 1.0/Re*(gamma_gas*kappa/Pr*(qFluctGrad(cell,qp,2,0)*wGradBF(cell,node,qp,0) + qFluctGrad(cell,qp,2,1)*wGradBF(cell,node,qp,1)))
                                     + force(cell,qp,2)*wBF(cell,node,qp); //rhobar*Tbar*(gamma-1)*div(u') + rhobar*ubar*dT'/dx + rhobar*vbar*dT'/dy +
                                                                           //1/Re*(d/dx(gamma*kappa/Pr*dT'/dx) + d/dy(gamma*kappa/Pr*dT'/dy) +
                                                                           //f2 
             Residual(cell, node, 3) += rhobar*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1))*wBF(cell,node,qp) 
                                     + ubar*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) 
                                     + force(cell,qp,3)*wBF(cell,node,qp); //rhobar*div(u') + ubar*drho'/dx + vbar*drho'/dy + f3 
                                     
            } 
          } 
        }
    }
    else if (numDims == 3) { //3D case
      double ubar = baseFlowData[0]; 
      double vbar = baseFlowData[1]; 
      double wbar = baseFlowData[2]; 
      double Tbar = baseFlowData[3]; 
      double rhobar = baseFlowData[4];
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++) 
             Residual(cell,node,i) = 0.0; 
          if(enableTransient)
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               for (std::size_t i=0; i < vecDim; i++) {
                  Residual(cell,node,i) = rhobar*qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
                  if (i == vecDim-1)
                    Residual(cell,node,i) = qFluctDot(cell,qp,i)*wBF(cell,node,qp); 
               }
          }
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell, node, 0) += rhobar*ubar*qFluctGrad(cell,qp,0,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,0,1)*wBF(cell,node,qp)
                                     + rhobar*wbar*qFluctGrad(cell,qp,0,2)*wBF(cell,node,qp) 
                                     + rhobar*Rgas*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) + Rgas*Tbar*qFluctGrad(cell,qp,4,0)*wBF(cell,node,qp) 
                                     + 1.0/Re*((2.0*mu + lambda)*qFluctGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + lambda*qFluctGrad(cell,qp,1,1)*wGradBF(cell,node,qp,0)
                                               + lambda*qFluctGrad(cell,qp,2,2)*wGradBF(cell,node,qp,0) 
                                               + mu*qFluctGrad(cell,qp,1,0)*wGradBF(cell,node,qp,1) + mu*qFluctGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1)
                                               + mu*qFluctGrad(cell,qp,2,0)*wGradBF(cell,node,qp,2) + mu*qFluctGrad(cell,qp,0,2)*wGradBF(cell,node,qp,2))
                                     + force(cell,qp,0)*wBF(cell,node,qp); //rhobar*ubar*du'/dx + rhobar*vbar*du'/dy + rhobar*wbar*du'/dz + rhobar*R*dT'/dx + R*Tbar*drho'/dx + 
                                                                           //1/Re*(d/dx((2*mu+lambda)*du'/dx + lambda*dv'/dy + lambda*dw'/dz) + d/dy*(mu*dv'/dx + mu*du'/dy)) + 
                                                                           //d/dz(mu*dw'/dx + mu*du'/dz)) + f0
             Residual(cell, node, 1) += rhobar*ubar*qFluctGrad(cell,qp,1,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,1,1)*wBF(cell,node,qp)
                                     + rhobar*wbar*qFluctGrad(cell,qp,1,2)*wBF(cell,node,qp)  
                                     + rhobar*Rgas*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) + Rgas*Tbar*qFluctGrad(cell,qp,4,1)*wBF(cell,node,qp) 
                                     + 1.0/Re*(mu*qFluctGrad(cell,qp,1,0)*wGradBF(cell,node,qp,0) + mu*qFluctGrad(cell,qp,0,1)*wGradBF(cell,node,qp,0)
                                               + lambda*qFluctGrad(cell,qp,0,0)*wGradBF(cell,node,qp,1) + (2.0*mu + lambda)*qFluctGrad(cell,qp,1,1)*wGradBF(cell,node,qp,1)
                                               + lambda*qFluctGrad(cell,qp,2,2)*wGradBF(cell,node,qp,1) 
                                               + mu*qFluctGrad(cell,qp,2,1)*wGradBF(cell,node,qp,2) + mu*qFluctGrad(cell,qp,1,2)*wGradBF(cell,node,qp,2))
                                     + force(cell,qp,1)*wBF(cell,node,qp); //rhobar*ubar*dv'/dx + rhobar*vbar*dv'/dy + rhobar*wbar*dv'/dz + rhobar*R*dT'/dy + R*Tbar*drho'/dy + 
                                                                           //1/Re*(d/dx(mu*dv'/dx + mu*du'/dy) + d/dy(lambda*du'/dx + (2*mu + lambda)*dv'/dy + lambda*dw'/dz) +
                                                                           //d/dz(mu*dw'/dy + mu*dv'/dz)) + f1
             Residual(cell, node, 2) += rhobar*ubar*qFluctGrad(cell,qp,2,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,2,1)*wBF(cell,node,qp) 
                                     + rhobar*wbar*qFluctGrad(cell,qp,2,2)*wBF(cell,node,qp) 
                                     + rhobar*Rgas*qFluctGrad(cell,qp,3,2)*wBF(cell,node,qp) + Rgas*Tbar*qFluctGrad(cell,qp,4,2)*wBF(cell,node,qp)
                                     + 1.0/Re*(mu*qFluctGrad(cell,qp,2,0)*wGradBF(cell,node,qp,0) + mu*qFluctGrad(cell,qp,0,2)*wGradBF(cell,node,qp,0) 
                                               + mu*qFluctGrad(cell,qp,2,1)*wGradBF(cell,node,qp,1) + mu*qFluctGrad(cell,qp,1,2)*wGradBF(cell,node,qp,1) 
                                               + lambda*qFluctGrad(cell,qp,0,0)*wGradBF(cell,node,qp,2) + lambda*qFluctGrad(cell,qp,1,1)*wGradBF(cell,node,qp,2) 
                                               + (2.0*mu + lambda)*qFluctGrad(cell,qp,2,2)*wGradBF(cell,node,qp,2)) 
                                     + force(cell,qp,2)*wBF(cell,node,qp); //rhobar*ubar*dw'/dx + rhobar*vbar*dw'/dy + rhobar*wbar*dw'/dz + rhobar*R*dT'/dz + R*Tbar*drho'/dz + 
                                                                           //1/Re*(d/dx(mu*dw'/dx + mu*du'/dz) + d/dy(mu*dw'/dy + mu*dv'/dz) + 
                                                                           //d/dz(lambda*du'/dx + lambda*dv'/dy + (2*mu + lambda)*dw'/dz)) + f2 
             Residual(cell, node, 3) += rhobar*Tbar*(gamma_gas - 1.0)*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1) + qFluctGrad(cell,qp,2,2))*wBF(cell,node,qp) 
                                     + rhobar*ubar*qFluctGrad(cell,qp,3,0)*wBF(cell,node,qp) + rhobar*vbar*qFluctGrad(cell,qp,3,1)*wBF(cell,node,qp) 
                                     + 1.0/Re*(gamma_gas*kappa/Pr*(qFluctGrad(cell,qp,3,0)*wGradBF(cell,node,qp,0) + qFluctGrad(cell,qp,3,1)*wGradBF(cell,node,qp,1) + qFluctGrad(cell,qp,3,2)*wGradBF(cell,node,qp,2)))
                                     + force(cell,qp,3)*wBF(cell,node,qp); //rhobar*Tbar*(gamma-1)*div(u') + rhobar*ubar*dT'/dx + rhobar*vbar*dT'/dy + rhobar*wbar*dT'/dz +
                                                                           //1/Re*(d/dx(gamma*kappa/Pr*dT'/dx) + d/dy(gamma*kappa/Pr*dT'/dy) + d/dz(gamma*kappa/Pr*dT'/dz +
                                                                           //f3
             Residual(cell, node, 4) += rhobar*(qFluctGrad(cell,qp,0,0) + qFluctGrad(cell,qp,1,1) + qFluctGrad(cell,qp,2,2))*wBF(cell,node,qp) 
                                     + ubar*qFluctGrad(cell,qp,4,0)*wBF(cell,node,qp) + vbar*qFluctGrad(cell,qp,4,1)*wBF(cell,node,qp) + wbar*qFluctGrad(cell,qp,4,2)*wBF(cell,node,qp)
                                     + force(cell,qp,4)*wBF(cell,node,qp); //rhobar*div(u') + ubar*drho'/dx + vbar*drho'/dy + wbar*drho'/dz + f4 
                                     
            } 
          } 
        }
    }
  }
}

//**********************************************************************
}

