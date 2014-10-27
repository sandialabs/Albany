//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
ScaleVector<EvalT, Traits>::
ScaleVector(const Teuchos::ParameterList& p) :
  inVector         (p.get<std::string>                   ("Input Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  outVector        (p.get<std::string>                   ("Output Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  coefficient      (p.get<double>                        ("Coefficient"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(inVector);
  this->addEvaluatedField(outVector);

  this->setName("ScaleVector"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScaleVector<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(outVector,fm);
  this->utils.setFieldData(inVector,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScaleVector<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      for (std::size_t i=0; i < numDims; ++i)
        outVector(cell,qp,i) = coefficient* inVector(cell,qp,i);
}

//**********************************************************************
}

