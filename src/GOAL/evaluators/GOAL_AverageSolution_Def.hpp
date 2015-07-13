//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Intrepid_MiniTensor_Mechanics.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>

namespace GOAL {

template<typename EvalT, typename Traits>
AverageSolution<EvalT, Traits>::
AverageSolution(
    Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  avgSolution(p.get<std::string>("QoI Name"), dl->node_vector)
{
  this->addDependentField(wBF);
  this->addEvaluatedField(avgSolution);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];
  numQPs = dims[2];
  numDims = dims[3];

  this->setName("AverageSolution" + PHX::typeAsString<EvalT>());
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void AverageSolution<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(avgSolution, fm);
}

template<typename EvalT, typename Traits>
void AverageSolution<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node)
    for (int dim = 0; dim < numDims; ++dim)
      avgSolution(cell, node, dim) = ScalarT(0);
    for (int pt = 0; pt < numQPs; ++pt)
    for (int node = 0; node < numNodes; ++node)
    for (int i = 0; i < numDims; ++i)
      avgSolution(cell, node, i) += wBF(cell, node, pt);
  }
}

}
