//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
CurrentCoords<EvalT, Traits>::CurrentCoords(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : refCoords(
          p.get<std::string>("Reference Coordinates Name"),
          dl->vertices_vector),
      displacement(p.get<std::string>("Displacement Name"), dl->node_vector),
      currentCoords(
          p.get<std::string>("Current Coordinates Name"),
          dl->node_vector)
{
  this->addDependentField(refCoords);
  this->addDependentField(displacement);

  this->addEvaluatedField(currentCoords);

  this->setName("Current Coordinates" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numDims     = dims[2];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
CurrentCoords<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(refCoords, fm);
  this->utils.setFieldData(displacement, fm);
  this->utils.setFieldData(currentCoords, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
CurrentCoords<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  for (int cell = 0; cell < workset.numCells; ++cell)
    for (int node = 0; node < numNodes; ++node)
      for (int dim = 0; dim < numDims; ++dim)
        currentCoords(cell, node, dim) =
            refCoords(cell, node, dim) + displacement(cell, node, dim);
}

//**********************************************************************
}  // namespace LCM
