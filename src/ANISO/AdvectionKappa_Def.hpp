//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "ANISO_Expression.hpp"

namespace ANISO {

template<typename EvalT, typename Traits>
AdvectionKappa<EvalT, Traits>::
AdvectionKappa(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  coord     (p.get<std::string>("Coordinate Name"), dl->qp_vector),
  kappa     (p.get<std::string>("Kappa Name"), dl->qp_scalar),
  kappa_val (p.get<std::string>("Kappa Value")) {

  num_qps = dl->node_qp_vector->dimension(2);
  num_dims = dl->node_qp_vector->dimension(3);

  this->addDependentField(coord);
  this->addEvaluatedField(kappa);
  this->setName("Advection Kappa");
}

template<typename EvalT, typename Traits>
void AdvectionKappa<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {

  this->utils.setFieldData(coord, fm);
  this->utils.setFieldData(kappa, fm);
}

template<typename EvalT, typename Traits>
void AdvectionKappa<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < num_qps; ++qp) {

      kappa(cell, qp) = expression_eval(
          kappa_val,
          coord(cell,qp,0),
          coord(cell,qp,1),
          coord(cell,qp,2),
          0);

    }
  }

}

}
