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
  wBF              (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF          (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  EtaUE            (p.get<std::string> ("EtaUE QP Variable Name"), dl->qp_vector),
  EtaUEDot         (p.get<std::string> ("EtaUE Dot QP Variable Name"), dl->qp_vector),
  EtaUEGrad        (p.get<std::string> ("EtaUE Gradient QP Variable Name"), dl->qp_vecgradient),
  EtaUEDotGrad     (p.get<std::string> ("EtaUE Dot Gradient QP Variable Name"), dl->qp_vecgradient),
  out              (Teuchos::VerboseObjectBase::getDefaultOStream()),
  waterDepthQP     (p.get<std::string> ("Water Depth QP Name"), dl->qp_scalar), 
  betaQP           (p.get<std::string> ("Beta QP Name"), dl->qp_scalar), 
  zalphaQP         (p.get<std::string> ("z_alpha QP Name"), dl->qp_scalar), 
  muSqr            (p.get<double>("Mu Squared")), 
  epsilon          (p.get<double>("Epsilon")), 
  force            (p.get<std::string> ("Body Force Name"), dl->qp_vector),
  waterDepthGrad   (p.get<std::string>("Water Depth Gradient Name"), dl->qp_gradient),
  Residual         (p.get<std::string> ("Residual Name"), dl->node_vector)
{

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(EtaUE);
  this->addDependentField(EtaUEDot);
  this->addDependentField(EtaUEGrad);
  this->addDependentField(EtaUEDotGrad);
  this->addDependentField(waterDepthQP);
  this->addDependentField(betaQP);
  this->addDependentField(zalphaQP);
  this->addDependentField(waterDepthGrad);
  this->addDependentField(force);

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
  this->utils.setFieldData(waterDepthQP,fm);
  this->utils.setFieldData(betaQP,fm);
  this->utils.setFieldData(zalphaQP,fm);
  this->utils.setFieldData(waterDepthGrad,fm);
  this->utils.setFieldData(force,fm);
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


  //IKT, FIXME, Zhiheng and Xiaoshu: fill in correctly!
  //ZW: Dimensional residuals
  if (vecDim == 3) {
    //1D case
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int qp=0; qp < numQPs; ++qp) {
            Residual(cell,node,0) += EtaUEDot(cell,qp,0)*wBF(cell,node,qp)
                                    + force(cell,qp,0)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,0)*(waterDepthQP(cell,qp)+EtaUE(cell,qp,0))*EtaUE(cell,qp,1)
                                    - wGradBF(cell,node,qp,0)*((0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp) - waterDepthQP(cell,qp)*waterDepthQP(cell,qp)/6)*waterDepthQP(cell,qp) + (zalphaQP(cell,qp)+0.5*waterDepthQP(cell,qp))*waterDepthQP(cell,qp)*waterDepthQP(cell,qp))*EtaUE(cell,qp,2);
            Residual(cell,node,1) += EtaUEDot(cell,qp,1)*wBF(cell,node,qp)
                                    + force(cell,qp,1)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,0)*(0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp)+zalphaQP(cell,qp)*waterDepthQP(cell,qp))*EtaUEDotGrad(cell,qp,1,0)
                                    - wGradBF(cell,node,qp,0)*(9.8*EtaUE(cell,qp,0)+EtaUE(cell,qp,1)*EtaUE(cell,qp,1));
            Residual(cell,node,2) += EtaUEDot(cell,qp,2)*wBF(cell,node,qp)
                                    + force(cell,qp,2)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,0)*EtaUEGrad(cell,qp,1,0); 
        }
      }
    }
  }
  else if (vecDim == 5) {
    //2D case
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (std::size_t j=0; j < numDims; ++j) { 
              Residual(cell,node,0) += EtaUEDot(cell,qp,0)*wBF(cell,node,qp)
                                    + force(cell,qp,0)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,j)*(waterDepthQP(cell,qp)+EtaUE(cell,qp,0))*EtaUE(cell,qp,j+1)
                                    - wGradBF(cell,node,qp,j)*((0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp) - waterDepthQP(cell,qp)*waterDepthQP(cell,qp)/6)*waterDepthQP(cell,qp) + (zalphaQP(cell,qp)+0.5*waterDepthQP(cell,qp))*waterDepthQP(cell,qp)*waterDepthQP(cell,qp))*EtaUE(cell,qp,j+3);
              Residual(cell,node,1) += EtaUEDot(cell,qp,1)*wBF(cell,node,qp)
                                    + force(cell,qp,1)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,0)*(0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp)+zalphaQP(cell,qp)*waterDepthQP(cell,qp))*EtaUEDotGrad(cell,qp,1,0)
                                    - wGradBF(cell,node,qp,0)*(9.8*EtaUE(cell,qp,0)+(EtaUE(cell,qp,1)*EtaUE(cell,qp,1) + EtaUE(cell,qp,2)*EtaUE(cell,qp,2))+0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp)*EtaUEDotGrad(cell,qp,2,1)+zalphaQP(cell,qp)*EtaUEDotGrad(cell,qp,2,1)*waterDepthQP(cell,qp));
              Residual(cell,node,2) += EtaUEDot(cell,qp,2)*wBF(cell,node,qp)
                                    + force(cell,qp,2)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,1)*(0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp)+zalphaQP(cell,qp)*waterDepthQP(cell,qp))*EtaUEDotGrad(cell,qp,2,1)
                                    - wGradBF(cell,node,qp,1)*(9.8*EtaUE(cell,qp,0)+(EtaUE(cell,qp,1)*EtaUE(cell,qp,1) + EtaUE(cell,qp,2)*EtaUE(cell,qp,2))+0.5*zalphaQP(cell,qp)*zalphaQP(cell,qp)*EtaUEDotGrad(cell,qp,1,0)+zalphaQP(cell,qp)*EtaUEDotGrad(cell,qp,1,0)*waterDepthQP(cell,qp));
              Residual(cell,node,3) += EtaUEDot(cell,qp,3)*wBF(cell,node,qp)
                                    + force(cell,qp,3)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,0)*(EtaUEGrad(cell,qp,1,0)+EtaUEGrad(cell,qp,2,1));
              Residual(cell,node,4) += EtaUEDot(cell,qp,4)*wBF(cell,node,qp)
                                    + force(cell,qp,3)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,1)*(EtaUEGrad(cell,qp,1,0)+EtaUEGrad(cell,qp,2,1));
          }
        }
      }
    }
   
  }
}

}

/*
// Non-dimensional residuals
  //IKT, FIXME, Zhiheng and Xiaoshu: fill in correctly!
  if (vecDim == 3) {
    //1D case
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int qp=0; qp < numQPs; ++qp) {
            Residual(cell,node,0) += EtaUEDot(cell,qp,0)*wBF(cell,node,qp)
                                    + force(cell,qp,0)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,0)*((waterDepthQP(cell,qp)+epsilon*EtaUE(cell,qp,0))*EtaUE(cell,qp,1)
                                    + muSqr*((betaQP(cell,qp)*betaQP(cell,qp)*0.5+betaQP(cell,qp)+1/3)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUE(cell,qp,2)));
            Residual(cell,node,1) += EtaUEDot(cell,qp,1)*wBF(cell,node,qp)
                                    + force(cell,qp,1)*wBF(cell,node,qp)
                                    - muSqr*wGradBF(cell,node,qp,0)*betaQP(cell,qp)*(betaQP(cell,qp)*0.5+1)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,1,0)
                                    - wGradBF(cell,node,qp,0)*(EtaUE(cell,qp,0)+0.5*epsilon*EtaUE(cell,qp,1)*EtaUE(cell,qp,1));       
            Residual(cell,node,2) += EtaUEDot(cell,qp,2)*wBF(cell,node,qp)
                                    + force(cell,qp,2)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,0)*EtaUEGrad(cell,qp,1,0); 
        } 
      }
    }
  }
  else if (vecDim == 5) {
    //2D case
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (std::size_t j=0; j < numDims; ++j) { 
              Residual(cell,node,0) += EtaUEDot(cell,qp,0)*wBF(cell,node,qp) 
                                    + force(cell,qp,0)*wBF(cell,node,qp)
                                    - wGradBF(cell,node,qp,j)*((waterDepthQP(cell,qp)+epsilon*EtaUE(cell,qp,0))*EtaUE(cell,qp,j+1)
                                    + muSqr*((betaQP(cell,qp)*betaQP(cell,qp)*0.5+betaQP(cell,qp)+1/3)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUE(cell,qp,j+3)));
              Residual(cell,node,1) += EtaUEDot(cell,qp,1)*wBF(cell,node,qp)
                                    + force(cell,qp,1)*wBF(cell,node,qp)
                                    - muSqr*wGradBF(cell,node,qp,0)*betaQP(cell,qp)*(betaQP(cell,qp)*0.5+1)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,1,0)
                                    - wGradBF(cell,node,qp,0)*(EtaUE(cell,qp,0)+0.5*epsilon*(EtaUE(cell,qp,1)*EtaUE(cell,qp,1) + EtaUE(cell,qp,2)*EtaUE(cell,qp,2))+muSqr*(0.5*betaQP(cell,qp)*betaQP(cell,qp)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,2,1)+betaQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,2,1)*waterDepthQP(cell,qp)));
              Residual(cell,node,2) += EtaUEDot(cell,qp,2)*wBF(cell,node,qp)
                                    + force(cell,qp,2)*wBF(cell,node,qp)
                                    - muSqr*wGradBF(cell,node,qp,1)*betaQP(cell,qp)*(betaQP(cell,qp)*0.5+1)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,2,1)
                                    - wGradBF(cell,node,qp,1)*(EtaUE(cell,qp,0)+0.5*epsilon*(EtaUE(cell,qp,1)*EtaUE(cell,qp,1) + EtaUE(cell,qp,2)*EtaUE(cell,qp,2))+muSqr*(0.5*betaQP(cell,qp)*betaQP(cell,qp)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,1,1)+betaQP(cell,qp)*waterDepthQP(cell,qp)*EtaUEDotGrad(cell,qp,1,0)*waterDepthQP(cell,qp)));
              Residual(cell,node,3) += EtaUEDot(cell,qp,3)*wBF(cell,node,qp)
                                    + force(cell,qp,3)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,0)*(EtaUEGrad(cell,qp,1,0)+EtaUEGrad(cell,qp,2,1));
              Residual(cell,node,4) += EtaUEDot(cell,qp,4)*wBF(cell,node,qp)
                                    + force(cell,qp,3)*wBF(cell,node,qp)
                                    + wGradBF(cell,node,qp,1)*(EtaUEGrad(cell,qp,1,0)+EtaUEGrad(cell,qp,2,1));
          }
        }
      }
    }
    
  }
}

}
*/
