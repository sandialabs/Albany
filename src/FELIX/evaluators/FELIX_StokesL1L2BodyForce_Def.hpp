//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace FELIX {
const double pi = 3.1415926535897932385;
const double g = 9.8; //gravity for FELIX; hard-coded here for now
const double rho = 910; //density for FELIX; hard-coded here for now

//**********************************************************************

template<typename EvalT, typename Traits>
StokesL1L2BodyForce<EvalT, Traits>::
StokesL1L2BodyForce(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  force(p.get<std::string>("Body Force Name"), dl->qp_vector ), 
  A(1.0), 
  n(3.0), 
  alpha(0.0)
{
  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  A = bf_list->get("Glen's Law A", 1.0); 
  n = bf_list->get("Glen's Law n", 3.0); 
  alpha = bf_list->get("FELIX alpha", 0.0);
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "L1L2SinCos") {
    bf_type = L1L2_SINCOS;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar); 
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->qp_vector->dimensions(dims);
  vecDim  = dims[2];

//  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
//*out << " in FELIX Stokes L1L2 source! " << endl;
//*out << " vecDim = " << vecDim << endl;
//*out << " numDims = " << numDims << endl;
//*out << " numQPs = " << numQPs << endl; 


  this->setName("StokesL1L2BodyForce"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesL1L2BodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(force,fm); 
  if (bf_type == L1L2_SINCOS) {
    this->utils.setFieldData(muFELIX,fm);
    this->utils.setFieldData(coordVec,fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesL1L2BodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < vecDim; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 else if (bf_type == L1L2_SINCOS) {
   double xphase=0.0, yphase=0.0;
   double r = 3*pi;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;  
       MeshScalarT muqp = pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase); 
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase); 
       MeshScalarT exx = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r; 
       MeshScalarT eyy = -2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) - r; 
       MeshScalarT exy = 0.0;  
       force(cell,qp,0) = 2.0*muqp*(-4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase))  
            + 2.0*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy); 
       force(cell,qp,1) = 2.0*muqp*(4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase)) 
            + 2.0*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }
   }
 }
}
}

