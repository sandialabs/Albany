//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
FixedFieldTerm<EvalT, Traits>::
FixedFieldTerm(const Teuchos::ParameterList& p) :
  fieldVal         (p.get<std::string>                   ("Field Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") ),
  outScalar        (p.get<std::string>                   ("Output Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout") )
{
  // Pull out numQPs from a Layout
  Teuchos::RCP<PHX::DataLayout> dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");
  std::vector<PHX::Device::size_type> dims;
  dl->dimensions(dims);
  numQPs  = dims[1];

  fixedValue = p.get<double>("Fixed Value");
  penaltyValue = p.get<double>("Penalty Coefficient");
  
  this->addDependentField(fieldVal);
  this->addEvaluatedField(outScalar);

  this->setName("FixedFieldTerm"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void FixedFieldTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(fieldVal,fm);
  this->utils.setFieldData(outScalar,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void FixedFieldTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      outScalar(cell,qp) = 1.0/2.0*penaltyValue
                           *(fieldVal(cell,qp)-fixedValue);
}

//**********************************************************************
}

