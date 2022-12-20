//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
JouleHeating<EvalT, Traits>::
JouleHeating(Teuchos::ParameterList& p) :
  potentialGrad(p.get<std::string>("Gradient Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  potentialFlux(p.get<std::string>("Flux Variable Name"),
		p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  jouleHeating(p.get<std::string>("Source Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addEvaluatedField(jouleHeating);
  this->addDependentField(potentialGrad.fieldTag());
  this->addDependentField(potentialFlux.fieldTag());
  this->setName("Joule Heating" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void JouleHeating<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(jouleHeating,fm);
  this->utils.setFieldData(potentialGrad,fm);
  this->utils.setFieldData(potentialFlux,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void JouleHeating<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Intrepid2::FunctionSpaceTools<PHX::Device>::dotMultiplyDataData
    (jouleHeating.get_view(), potentialFlux.get_view(), potentialGrad.get_view());
}
// **********************************************************************
// **********************************************************************
}
