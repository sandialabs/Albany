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
  else if (type == "Constant") {
    bf_type = CONSTANT;
    rho = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("Density QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(rho);
  }
  else if (type == "Poly") {
    bf_type = POLY;  
    mu = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("Material Property Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(mu); 
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  if (bf_type == CONSTANT) {
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
  if (bf_type == CONSTANT) {
    this->utils.setFieldData(rho,fm);
  }
  else if (bf_type == POLY) {
    this->utils.setFieldData(mu,fm);
    this->utils.setFieldData(coordVec,fm);
  }

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 for (std::size_t cell=0; cell < workset.numCells; ++cell) {
   for (std::size_t qp=0; qp < numQPs; ++qp) {      
     for (std::size_t i=0; i < numDims; ++i) {
       if (bf_type == NONE)
	 force(cell,qp,i) = 0.0;
       else if (bf_type == CONSTANT) {
	 force(cell,qp,i) = rho(cell,qp)*gravity[i];
       }
     }
     //the following is hard-coded for a 2D Stokes problem with manufactured solution
     if (bf_type == POLY) {
       //double *X = &coordVec(cell, qp, 0);
       //double *Y = &coordVec(cell, qp, 1);    
       //f[0] = -40*mu*(2*y^2 - 3*y + 1)*y*(6*x^2 - 6*x + 1) - 120*mu*(x - 1)^2*x^2*(2*y - 1) - 10*mu; 
       force(cell, qp, 0) = -40.0*mu(cell,qp)*(2.0*coordVec(cell,qp,1)*coordVec(cell,qp,1) - 3.0*coordVec(cell,qp,1)+1.0)*coordVec(cell,qp,1)*(6.0*coordVec(cell,qp,0)*coordVec(cell,qp,0) 
                             -6.0*coordVec(cell,qp,0) + 1.0) - 120*mu(cell,qp)*(coordVec(cell,qp,0)-1.0)*(coordVec(cell,qp,0)-1.0)*coordVec(cell,qp,0)*coordVec(cell,qp,0)*(2.0*coordVec(cell,qp,1)-1.0) 
                             - 10.0*mu(cell,qp);      
       force(cell, qp, 0) = -1.0*force(cell, qp, 0);  
       //f[1] = 120*mu*(1 - y)^2*y^2*(2*x-1) + 40*mu*(2*x^2 - 3*x + 1)*x*(6*y^2 - 6*y + 1) + 5*mu*y; 
       force(cell, qp, 1) = 120.0*mu(cell,qp)*(1.0-coordVec(cell,qp,1))*(1.0-coordVec(cell,qp,1))*coordVec(cell,qp,1)*coordVec(cell,qp,1)*(2.0*coordVec(cell,qp,0)-1.0) + 
                            40.0*mu(cell,qp)*(2.0*coordVec(cell,qp,0)*coordVec(cell,qp,0) - 3.0*coordVec(cell,qp,0) + 1.0)*coordVec(cell,qp,0)*(6.0*coordVec(cell,qp,1)*coordVec(cell,qp,1) - 
                            6.0*coordVec(cell,qp,1) + 1.0) + 5*mu(cell,qp)*coordVec(cell,qp,1);
       force(cell, qp, 1) = -1.0*force(cell, qp, 1);  
     }
   }
 }
}

}

