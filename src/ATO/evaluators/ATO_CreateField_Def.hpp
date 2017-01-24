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
CreateField<EvalT, Traits>::
CreateField(const Teuchos::ParameterList& p) :
  field          (p.get<std::string>                   ("Field Name"),
                  p.get<Teuchos::RCP<PHX::DataLayout> >("Field Data Layout") )
{
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Field Data Layout");
  std::vector<PHX::Device::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];

  value = p.get<double>("Field Value");

  this->addEvaluatedField(field);

  this->setName("CreateField"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CreateField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CreateField<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
        field(cell,qp) = value;
}

//**********************************************************************
}

