//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
PeridigmForce<EvalT, Traits>::
PeridigmForce(const Teuchos::ParameterList& p) :
  referenceCoordinates  (p.get<std::string>                   ("Reference Coordinates Name"),
                          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  currentCoordinates    (p.get<std::string>                   ("Current Coordinates Name"),
                          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  force                 (p.get<std::string>                   ("Force Name"),
                         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )

{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(referenceCoordinates);
  this->addDependentField(currentCoordinates);

  this->addEvaluatedField(force);

  this->setName("Peridigm"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(referenceCoordinates,fm);
  this->utils.setFieldData(currentCoordinates,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // 1)  Copy from referenceCoordinates and currentCoordinates fields into Epetra_Vectors for Peridigm

  // 2)  Call Peridigm

  // 3)  Copy nodal forces from Epetra_Vector to multi-dimensional arrays

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      force(cell,qp,0) = 0.0;
      force(cell,qp,1) = 0.0;
      force(cell,qp,2) = 0.0;
    }
  }
}

//**********************************************************************
}

