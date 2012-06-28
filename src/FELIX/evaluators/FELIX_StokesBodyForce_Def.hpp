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
       else if (bf_type == CONSTANT)
	 force(cell,qp,i) = rho(cell,qp)*gravity[i];
     }
   }
 }
}

}

