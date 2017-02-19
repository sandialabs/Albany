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
NSBodyForce<EvalT, Traits>::
NSBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  haveHeat(p.get<bool>("Have Heat"))
{

  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Constant") {
    bf_type = CONSTANT;
    rho = decltype(rho)(
            p.get<std::string>("Density QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(rho);
  }
  else if (type == "Boussinesq") {
    TEUCHOS_TEST_FOR_EXCEPTION(haveHeat == false, std::logic_error,
		       std::endl <<
		       "Error!  Must enable heat equation for Boussinesq " <<
		       "body force term!");
    bf_type = BOUSSINESQ;
    T = decltype(T)(
          p.get<std::string>("Temperature QP Variable Name"),
	  p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    rho = decltype(rho)(
            p.get<std::string>("Density QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    beta = decltype(beta)(
            p.get<std::string>(
              "Volumetric Expansion Coefficient QP Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(rho);
    this->addDependentField(beta);
    this->addDependentField(T);
  }

  this->addEvaluatedField(force);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  if (bf_type == CONSTANT || bf_type == BOUSSINESQ) {
    if (bf_list->isType<Teuchos::Array<double> >("Gravity Vector"))
      gravity = bf_list->get<Teuchos::Array<double> >("Gravity Vector");
    else {
      gravity.resize(numDims);
      gravity[numDims-1] = -1.0;
    }
  }
  this->setName("NSBodyForce" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == CONSTANT) {
    this->utils.setFieldData(rho,fm);
  }
  else if (bf_type == BOUSSINESQ) {
    this->utils.setFieldData(T,fm);
    this->utils.setFieldData(rho,fm);
    this->utils.setFieldData(beta,fm);
  }

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 for (std::size_t cell=0; cell < workset.numCells; ++cell) {
   for (std::size_t qp=0; qp < numQPs; ++qp) {      
     for (std::size_t i=0; i < numDims; ++i) {
       if (bf_type == NONE)
	 force(cell,qp,i) = 0.0;
       else if (bf_type == CONSTANT)
	 force(cell,qp,i) = rho(cell,qp)*gravity[i];
       else if (bf_type == BOUSSINESQ)
	 force(cell,qp,i) = rho(cell,qp)*T(cell,qp)*beta(cell,qp)*gravity[i];
     }
   }
 }
}

}

