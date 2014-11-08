//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>

#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  BifurcationCheck<EvalT, Traits>::
  BifurcationCheck(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    tangent_(p.get<std::string>("Material Tangent Name"),dl->qp_tensor4),
    ellipticity_flag_(p.get<std::string>("Ellipticity Flag Name"),dl->qp_scalar),
    direction_(p.get<std::string>("Bifurcation Direction Name"),dl->qp_vector)
  {
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    this->addDependentField(tangent_);

    this->addEvaluatedField(ellipticity_flag_);
    this->addEvaluatedField(direction_);

    this->setName("BifurcationCheck"+PHX::typeAsString<PHX::Device>());
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(tangent_,fm);
    this->utils.setFieldData(ellipticity_flag_,fm);
    this->utils.setFieldData(direction_,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    Intrepid::Vector<ScalarT> direction(1.0, 0.0, 0.0);
    Intrepid::Tensor4<ScalarT> tangent(num_dims_);

    // Compute DefGrad tensor from displacement gradient
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {

        tangent.fill( tangent_,cell,pt,-1,-1,-1) );
        ellipticity_flag_(cell,pt) = 0;

        // Intrepid::check_ellipticity(tan);

        for (std::size_t i(0); i < num_dims_; ++i) {
          direction_(cell,pt,i) = direction(i);
        }
      }
    }
  }
  //----------------------------------------------------------------------------
}
