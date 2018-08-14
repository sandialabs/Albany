//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>
#include <vector>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
GatherSphereVolume<EvalT, Traits>::GatherSphereVolume(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : sphereVolume(p.get<std::string>("Sphere Volume Name"), dl->node_scalar),
      numVertices(0),
      worksetSize(0)
{
  this->addEvaluatedField(sphereVolume);
  this->setName("Gather Sphere Volume" + PHX::typeAsString<EvalT>());
}

template <typename EvalT, typename Traits>
GatherSphereVolume<EvalT, Traits>::GatherSphereVolume(
    const Teuchos::ParameterList& p)
    : sphereVolume(
          p.get<std::string>("Sphere Volume Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Data Layout")),
      numVertices(0),
      worksetSize(0)
{
  this->addEvaluatedField(sphereVolume);
  this->setName("Gather Sphere Volume" + PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
GatherSphereVolume<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphereVolume, fm);

  typename std::vector<
      typename PHX::template MDField<ScalarT, Cell, Vertex>::size_type>
      dims;
  sphereVolume.dimensions(dims);  // get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
// **********************************************************************
template <typename EvalT, typename Traits>
void
GatherSphereVolume<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  unsigned int              numCells       = workset.numCells;
  Teuchos::ArrayRCP<double> wsSphereVolume = workset.wsSphereVolume;

  TEUCHOS_TEST_FOR_EXCEPTION(
      wsSphereVolume.is_null(),
      std::logic_error,
      "\n****Error:  Sphere Volume field not found in GatherSphereVolume "
      "evaluator!\n");

  for (int cell = 0; cell < numCells; ++cell) {
    for (int v = 0; v < sphereVolume.dimension(1); v++)
      sphereVolume(cell, v) = wsSphereVolume[cell, v];
  }

  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to calculations on singular elements.
  for (int cell = numCells; cell < worksetSize; ++cell) {
    for (int v = 0; v < sphereVolume.dimension(1); v++)
      sphereVolume(cell, v) = sphereVolume(0, v);
  }
}
#pragma clang diagnostic pop
}  // namespace LCM
