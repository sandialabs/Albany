//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {
const double pi = 3.1415926535897932385;
//**********************************************************************

template<typename EvalT, typename Traits>
ComprNSBodyForce<EvalT, Traits>::
ComprNSBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ) 
{
  std::cout << "Compr NS body force constructor!" << std::endl; 
  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Taylor-Green Vortex") {
    std::cout << "Taylor-Green Vortex source" << std::endl; 
    bf_type = TAYLOR_GREEN_VORTEX;  
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout") );
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  Teuchos::RCP<PHX::DataLayout> gradient_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  gradient_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  vector_dl->dimensions(dims);
  vecDim  = dims[2];

std::cout << " in Compr NS Stokes source! " << std::endl;
std::cout << " vecDim = " << vecDim << std::endl;
std::cout << " numDims = " << numDims << std::endl;
std::cout << " numQPs = " << numQPs << std::endl; 


  this->setName("ComprNSBodyForce" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == TAYLOR_GREEN_VORTEX) {
    this->utils.setFieldData(coordVec,fm);
  }
  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < vecDim; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 else if (bf_type == TAYLOR_GREEN_VORTEX) { //source term for MMS Taylor-Vortex-like problem 
   const RealType time = workset.current_time; //time
   const double Re = 1.0; 
   const double Pr = 0.72; 
   const double gamma_gas = 1.4; 
   const double kappa = 1.0; 
   const double mu = 1.0/Re; 
   const double Rgas = 0.714285733; //non-dimensional gas constant
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       force(cell,qp,0) = 0.0;
       force(cell,qp,1) = 2.0*exp(-2.0*time)*cos(x2pi)*(-sin(y2pi) + exp(-2.0*time)*sin(x2pi)*pi + 4.0*mu*pi*pi*sin(y2pi)) + 2.0*Rgas*pi*exp(-4.0*time)*sin(x2pi); 
       force(cell,qp,2) = 2.0*exp(-2.0*time)*cos(y2pi)*(sin(x2pi) + exp(-2.0*time)*sin(y2pi)*pi - 4.0*mu*pi*pi*sin(x2pi)) + 2.0*Rgas*pi*exp(-4.0*time)*sin(y2pi); 
       force(cell,qp,3) = -2.0*exp(-4.0*time)*(-2.0*cos(x2pi) - 2.0*cos(y2pi) + exp(-2.0*time)*cos(x2pi)*sin(y2pi)*sin(x2pi)*pi 
                                               - exp(-2.0*time)*sin(x2pi)*cos(y2pi)*sin(y2pi)*pi) 
         + (gamma_gas-1.0)/Rgas*4.0*mu*exp(-2.0*time)*sin(x2pi)*sin(y2pi)*pi*2.0*exp(-2.0*time)*sin(x2pi)*pi*sin(y2pi)  
         + (gamma_gas-1.0)/Rgas*4.0*mu*exp(-2.0*time)*sin(x2pi)*pi*sin(y2pi)*2.0*exp(-2.0*time)*sin(x2pi)*pi*sin(y2pi) 
         - gamma_gas*kappa/(Pr*Re)*4.0*exp(-4.0*time)*pi*pi*(cos(x2pi) + cos(y2pi));  
     }
   }
 }
}

}

