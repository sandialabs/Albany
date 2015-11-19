//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Application.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_PUMIDiscretization.hpp"
#include "PHAL_Workset.hpp"

namespace GOAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ScatterQoI<EvalT, Traits>::
ScatterQoI(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setName("ScatterQoI"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScatterQoI<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScatterQoI<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

//**********************************************************************
// Jacobian Specialization
//**********************************************************************
template<typename Traits>
ScatterQoI<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterQoI(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  qoi (p.get<std::string>("QoI Name"), dl->cell_scalar2)
{
  operation = Teuchos::rcp(new PHX::Tag<ScalarT>("Scatter QoI", dl->dummy));

  this->addDependentField(qoi);
  this->addEvaluatedField(*operation);

  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numNodes = dim[1];

  this->setName("ScatterQoI"+PHX::typeAsString<PHAL::AlbanyTraits::Jacobian>());
}

//**********************************************************************
template<typename Traits>
void ScatterQoI<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qoi, fm);
}

//**********************************************************************
template<typename Traits>
void ScatterQoI<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  const int neq = workset.wsElNodeEqID[0][0].size();
  const int nunk = neq*numNodes;

  for (int cell=0; cell < workset.numCells; ++cell) {
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > const& nodeID =
      workset.wsElNodeEqID[cell];
    int lunk=0;
    for (int node=0; node < numNodes; ++node) {
      for (int eq=0; eq < neq; ++eq) {
        const LO row = nodeID[node][eq];
        // scatter qoi(cell).fastAccessDx(lunk) into qoi vector
        lunk++;
      }
    }
  }
}

}
