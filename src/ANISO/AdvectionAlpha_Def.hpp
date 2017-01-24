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
AdvectionAlpha<EvalT, Traits>::
AdvectionAlpha(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  coord     (p.get<std::string>("Coordinate Name"), dl->qp_vector),
  alpha     (p.get<std::string>("Alpha Name"), dl->qp_vector),
  alpha_mag (p.get<std::string>("Alpha Magnitude Name"), dl->qp_scalar),
  alpha_val (p.get<Teuchos::Array<std::string> >("Alpha Value")) {

  num_qps = dl->node_qp_vector->dimension(2);
  num_dims = dl->node_qp_vector->dimension(3);

  this->addDependentField(coord);
  this->addEvaluatedField(alpha);
  this->addEvaluatedField(alpha_mag);
  this->setName("Advection Alpha");
}

template<typename EvalT, typename Traits>
void AdvectionAlpha<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {

  this->utils.setFieldData(coord, fm);
  this->utils.setFieldData(alpha, fm);
  this->utils.setFieldData(alpha_mag, fm);
}

template<typename EvalT, typename Traits>
void AdvectionAlpha<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < num_qps; ++qp) {
      alpha_mag(cell, qp) = 0.0;
      for (int dim=0; dim < num_dims; ++dim) {
        alpha(cell, qp, dim) = expression_eval(
            alpha_val[dim],
            coord(cell,qp,0),
            coord(cell,qp,1),
            coord(cell,qp,2),
            0);
        alpha_mag(cell, qp) += alpha(cell, qp, dim)*alpha(cell, qp, dim);
      }
      alpha_mag(cell, qp) = std::sqrt(alpha_mag(cell, qp));
    }
  }

}

}
