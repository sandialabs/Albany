//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {
const double pi = 3.1415926535897932385;
//**********************************************************************

template<typename EvalT, typename Traits>
LinComprNSBodyForce<EvalT, Traits>::
LinComprNSBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ) 
{
  std::cout << "Lin Compr NS body force constructor!" << std::endl; 
  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Steady Euler") {
    bf_type = STEADYEUL;  
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout") );
    this->addDependentField(coordVec);
  }
  else if (type == "Unsteady Euler MMS") {
    bf_type = UNSTEADYEULMMS;  
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout") );
    this->addDependentField(coordVec);
  }
  else if (type == "Driven Pulse") {
    bf_type = DRIVENPULSE;  
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

std::cout << " in Lin Compr NS Stokes source! " << std::endl;
std::cout << " vecDim = " << vecDim << std::endl;
std::cout << " numDims = " << numDims << std::endl;
std::cout << " numQPs = " << numQPs << std::endl; 


  this->setName("LinComprNSBodyForce"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == STEADYEUL || bf_type == UNSTEADYEULMMS || bf_type == DRIVENPULSE) {
    this->utils.setFieldData(coordVec,fm);
  }

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < vecDim; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 else if (bf_type == STEADYEUL) {
    const double ubar = 1.0; 
    const double vbar = 1.0; 
    const double zetabar = 1.0; 
    const double pbar = 0.0;
    const double gamma_gas = 1.4; 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT x = coordVec(cell,qp,0); 
       MeshScalarT y = coordVec(cell,qp,1); 
       f[0] = -1.0*(ubar*(y - x*sin(x)) + vbar*x + zetabar*2.0*x*(0.5-y));  
       f[1] = -1.0*(ubar*cos(x)*y + vbar*sin(x) - zetabar*x*x);  
       f[2] = -1.0*(gamma_gas*pbar*(y - x*sin(x) + sin(x)) + ubar*2.0*x*(0.5-y) - vbar*x*x);  
     }
   }
 }
 else if (bf_type == UNSTEADYEULMMS) {
   const double ubar = 0.0; 
   const double vbar = 0.0; 
   const double zetabar = 1.0; 
   const double pbar = 0.7142857;
   const double a = 1.0; 
   const double gamma_gas = 1.4; 
   const RealType time = workset.current_time;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT x = coordVec(cell,qp,0); 
       MeshScalarT y = coordVec(cell,qp,1);
       f[0] = -1.0*exp(-a*time)*(-a*sin(x2pi)*cos(y2pi) + ubar*2.0*pi*cos(x2pi)*cos(y2pi) 
                                 -vbar*2.0*pi*sin(x2pi)*sin(y2pi) + 2.0*pi*zetabar*cos(x2pi)*sin(y2pi));   
       f[1] = -1.0*exp(-a*time)*(-a*cos(x2pi)*sin(y2pi) - 2.0*pi*ubar*sin(x2pi)*sin(y2pi) 
                                 + vbar*2.0*pi*cos(x2pi)*cos(y2pi) + 2.0*pi*zetabar*sin(x2pi)*cos(y2pi)); 
       f[2] = -1.0*exp(-a*time)*(-a*sin(x2pi)*sin(y2pi) + gamma_gas*pbar*4.0*pi*cos(x2pi)*cos(y2pi) + 
                                 ubar*2.0*pi*cos(x2pi)*sin(y2pi) + vbar*2.0*pi*sin(x2pi)*cos(y2pi));   
     }
   }
 }
 else if (bf_type == DRIVENPULSE) {
   const RealType time = workset.current_time;
   const double tref = 1.0/347.9693;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0); 
       MeshScalarT y = coordVec(cell,qp,1);
       f[0] = 0.0; 
       if ((x >= 0.9) && (x <= 1.0) && (y >= 0.9) && (y <= 1.0))
         f[1] = (1.0e-4)*cos(2.0*pi*1000*time*tref); 
       else 
         f[1] = 0.0; 
       f[2] = 0.0; 
     }
   }

 }
}

}

