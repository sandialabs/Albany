//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {
const double pi = 3.1415926535897932385;

//**********************************************************************
template<typename EvalT, typename Traits>
StokesRm<EvalT, Traits>::
StokesRm(const Teuchos::ParameterList& p,
         const Teuchos::RCP<Albany::Layouts>& dl) :
  pGrad  (p.get<std::string> ("Pressure Gradient QP Variable Name"), dl->qp_vector),
  VGrad  (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_tensor),
  V      (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  force  (p.get<std::string> ("Body Force QP Variable Name"), dl->qp_vector),
  Rm     (p.get<std::string> ("Rm Name"), dl->qp_vector)
 
{
  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
  this->addDependentField(coordVec);

  this->addDependentField(pGrad);
  this->addDependentField(VGrad);
  this->addDependentField(V);
  this->addDependentField(force); 
  this->addEvaluatedField(Rm);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("StokesRm"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesRm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(pGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(force,fm);
  this->utils.setFieldData(coordVec,fm);

  this->utils.setFieldData(Rm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesRm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       double xphase=0.0, yphase=0.0; 
       double r = 3.0*pi; 
       double n = 3.0; 
       double A = 1.0; 
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;  
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase); 
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase);  
      for (std::size_t i=0; i < numDims; ++i) {
        Rm(cell,qp,i) = 0;
        Rm(cell,qp,i) += pGrad(cell,qp,i)+force(cell,qp,i); // - 0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*VGrad(cell,qp,i,0) + dmuargtdy*VGrad(cell,qp,i,1));
        //cout << force(cell, qp, i) << endl;  
      } 
    }
  }
}

}

