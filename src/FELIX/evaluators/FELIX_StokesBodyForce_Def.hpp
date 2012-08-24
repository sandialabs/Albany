/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {
const double pi = 3.1415926535897932385;
const double g = 9.8; //gravity for FELIX; hard-coded here for now
const double rho = 910; //density for FELIX; hard-coded here for now 


//**********************************************************************
template<typename EvalT, typename Traits>
StokesBodyForce<EvalT, Traits>::
StokesBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
{

  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Gravity") {
    bf_type = GRAVITY;
  }
  else if (type == "Poly") {
    bf_type = POLY;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "SinSin") {
    bf_type = SINSIN;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "SinCosZ") {
    bf_type = SINCOSZ;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  if (bf_type == GRAVITY) {
    if (bf_list->isType<Teuchos::Array<double> >("Gravity Vector"))
      gravity = bf_list->get<Teuchos::Array<double> >("Gravity Vector");
    else {
      gravity.resize(numDims);
      gravity[numDims-1] = -1.0;
    }
  }
  this->setName("StokesBodyForce"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == GRAVITY) {
  }
  else if (bf_type == POLY || bf_type == SINSIN || bf_type == SINCOSZ) {
    this->utils.setFieldData(muFELIX,fm);
    this->utils.setFieldData(coordVec,fm);
  }

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < numDims; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 else if (bf_type == GRAVITY) {
 for (std::size_t cell=0; cell < workset.numCells; ++cell) 
   for (std::size_t qp=0; qp < numQPs; ++qp) 
     for (std::size_t i=0; i < numDims; ++i) 
	 force(cell,qp,i) = -1.0*rho*g*gravity[i];
 }

 //The following is hard-coded for a 2D Stokes problem with manufactured solution
 else if (bf_type == POLY) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT* X = &coordVec(cell,qp,0);
       ScalarT& muqp = muFELIX(cell,qp);
       f[0] =  40.0*muqp*(2.0*X[1]*X[1] - 3.0*X[1]+1.0)*X[1]*(6.0*X[0]*X[0] -6.0*X[0] + 1.0)
              + 120*muqp*(X[0]-1.0)*(X[0]-1.0)*X[0]*X[0]*(2.0*X[1]-1.0) 
              + 10.0*muqp;      
       f[1] = - 120.0*muqp*(1.0-X[1])*(1.0-X[1])*X[1]*X[1]*(2.0*X[0]-1.0)
              - 40.0*muqp*(2.0*X[0]*X[0] - 3.0*X[0] + 1.0)*X[0]*(6.0*X[1]*X[1] - 6.0*X[1] + 1.0)
              - 5*muqp*X[1];
     }
   }
 }
 // Doubly-periodic MMS derived by Irina. 
 else if (bf_type == SINSIN) {
   double xphase=0.0, yphase=0.0; // Expose as parameters for verification
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       ScalarT& muqp = muFELIX(cell,qp);

       f[0] = -4.0*muqp*pi*(2*pi-1)*sin(x2pi + xphase)*sin(y2pi + yphase);
       f[1] = -4.0*muqp*pi*(2*pi+1)*cos(x2pi + xphase)*cos(y2pi + yphase);
     }
   }
 }
 // Doubly-periodic MMS with polynomial in Z
 else if (bf_type == SINCOSZ) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT& z = coordVec(cell,qp,2);
       ScalarT& muqp = muFELIX(cell,qp);

       /*ScalarT t1 = muqp*z*(1-z)*(1-2*z);
       ScalarT t2 = muqp*(1-6*z+6*z*z);
       ScalarT t3 = muqp*z*z*(1-z)*(1-z);

       f[0] =  (-8*pi*pi*t1 + t2 + 4*pi*pi)*sin(x2pi)*sin(y2pi);
       f[1] = -(-8*pi*pi*t1 + t2 + 4*pi*pi)*cos(x2pi)*cos(y2pi);
       f[2] = -2*pi*(-8*pi*pi*t3 + 2*t1 + 1)*cos(x2pi)*sin(y2pi);
       */
       ScalarT t1 = muqp*z*(1.0-z)*(1.0-2.0*z); 
       ScalarT t2 = muqp*(12.0*z - 6.0); 
       ScalarT t3 = muqp*z*z*(1.0-z)*(1.0-z); 
       ScalarT t4 = muqp*(1.0-6.0*z+6.0*z*z); 
     
       f[0] = (-8.0*pi*pi*t1 + t2 + 4.0*pi*pi*z)*sin(x2pi)*sin(y2pi); 
       f[1] = (8.0*pi*pi*t1 - t2 - 4.0*pi*pi*z)*cos(x2pi)*cos(y2pi); 
       f[2] = (16.0*pi*pi*pi*t3 - 4.0*pi*t4 - 2.0*pi)*cos(x2pi)*sin(y2pi); 
     }
   }
 }
}

}

