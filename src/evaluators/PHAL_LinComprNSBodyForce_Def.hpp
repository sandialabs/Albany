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

//base flow values
const double ubar = 0.0; 
const double vbar = 0.0; 
const double wbar = 0.0; 
const double zetabar = 1.0; 
const double pbar = 0.714285714285714;
//fluid parameters  
const double alpha = 1.0; 
const double gamma_gas = 1.4; //gas constant 

//**********************************************************************

template<typename EvalT, typename Traits>
LinComprNSBodyForce<EvalT, Traits>::
LinComprNSBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ) 
{
  cout << "Lin Compr NS body force constructor!" << endl; 
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

cout << " in Lin Compr NS Stokes source! " << endl;
cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;
cout << " numQPs = " << numQPs << endl; 


  this->setName("LinComprNSBodyForce"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == STEADYEUL) {
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
    double ubar = 1.0; 
    double vbar = 1.0; 
    double zetabar = 1.0; 
    double pbar = 0.0;
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
}

}

