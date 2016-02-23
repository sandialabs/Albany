//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Intrepid2_MiniTensor.h>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  UnitGradient<EvalT, Traits>::
  UnitGradient(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
    scalar_grad_(p.get<std::string>("Gradient QP Variable Name"), dl->qp_vector),
    unit_grad_(p.get<std::string>("Unit Gradient QP Variable Name"),dl->qp_vector)
  {
    this->addDependentField(scalar_grad_);
    this->addEvaluatedField(unit_grad_);
    this->setName("Unit Gradient QP Variable"+PHX::typeAsString<EvalT>());

    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_vector->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void UnitGradient<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalar_grad_,fm);
    this->utils.setFieldData(unit_grad_,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void UnitGradient<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    ScalarT scalar_mag(0.0);
    Intrepid2::Vector<ScalarT> grad(num_dims_), unit(num_dims_);

    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        grad.fill( scalar_grad_, cell,pt,0 );
        scalar_mag = Intrepid2::norm(grad);
        if (scalar_mag > 0) {
          unit = grad / scalar_mag;
          for (int i(0); i < num_dims_; i++) {
            unit_grad_(cell,pt,i) = unit(i);
          }
        }
        else {
          for (int i(0); i < num_dims_; i++){
            unit_grad_(cell,pt,i) = 1/std::sqrt(num_dims_);
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
}
