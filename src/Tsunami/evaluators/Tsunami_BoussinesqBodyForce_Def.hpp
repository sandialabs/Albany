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
BoussinesqBodyForce<EvalT, Traits>::
BoussinesqBodyForce(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  out              (Teuchos::VerboseObjectBase::getDefaultOStream()),
  waterDepthQP     (p.get<std::string> ("Water Depth QP Name"), dl->qp_scalar), 
  betaQP           (p.get<std::string> ("Beta QP Name"), dl->qp_scalar), 
  zalphaQP         (p.get<std::string> ("z_alpha QP Name"), dl->qp_scalar), 
  muSqr            (p.get<double>("Mu Squared")), 
  epsilon          (p.get<double>("Epsilon")), 
  force            (p.get<std::string> ("Body Force Name"), dl->qp_vector)
{

  this->addDependentField(waterDepthQP);
  this->addDependentField(betaQP);
  this->addDependentField(zalphaQP);

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  force.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];
   
 
  Teuchos::ParameterList* bf_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  
  if (type == "None") {
    bf_type = NONE;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Error in Tsunami::Boussinesq: Invalid Body Force Type = "
            << type << "!  Valid types are: 'None'.");
  }

  this->setName("BoussinesqBodyForce"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(waterDepthQP,fm);
  this->utils.setFieldData(betaQP,fm);
  this->utils.setFieldData(zalphaQP,fm);
  this->utils.setFieldData(force,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Zero out body force
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int qp=0; qp < numQPs; ++qp) 
      for (int i=0; i<vecDim; i++) 
          force(cell,qp,i) = 0.0; 

}

}

