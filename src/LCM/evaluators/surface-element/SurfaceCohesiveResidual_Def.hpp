//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceCohesiveResidual<EvalT, Traits>::SurfaceCohesiveResidual(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : cubature_(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepid_basis_(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      ref_area_(p.get<std::string>("Reference Area Name"), dl->qp_scalar),
      cohesive_traction_(
          p.get<std::string>("Cohesive Traction Name"),
          dl->qp_vector),
      force_(
          p.get<std::string>("Surface Cohesive Residual Name"),
          dl->node_vector)
{
  this->addDependentField(ref_area_);
  this->addDependentField(cohesive_traction_);

  this->addEvaluatedField(force_);

  this->setName("Surface Cohesive Residual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_vector->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_dims_     = dims[2];

  dl->qp_vector->dimensions(dims);

  num_qps_ = cubature_->getNumPoints();

  num_surf_nodes_ = num_nodes_ / 2;
  num_surf_dims_  = num_dims_ - 1;
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceCohesiveResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(cohesive_traction_, fm);
  this->utils.setFieldData(ref_area_, fm);
  this->utils.setFieldData(force_, fm);

  // Allocate Temporary Views
  ref_values_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_surf_nodes_, num_qps_);
  ref_grads_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_surf_nodes_, num_qps_, num_surf_dims_);
  ref_points_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_qps_, num_surf_dims_);
  ref_weights_ = Kokkos::DynRankView<RealType, PHX::Device>("XXX", num_qps_);

  // Pre-Calculate reference element quantitites
  cubature_->getCubature(ref_points_, ref_weights_);
  intrepid_basis_->getValues(
      ref_values_, ref_points_, Intrepid2::OPERATOR_VALUE);

  intrepid_basis_->getValues(ref_grads_, ref_points_, Intrepid2::OPERATOR_GRAD);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceCohesiveResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Vector<ScalarT> f_plus(0, 0, 0);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int bottom_node(0); bottom_node < num_surf_nodes_; ++bottom_node) {
      int top_node = bottom_node + num_surf_nodes_;

      // initialize force vector
      f_plus.fill(minitensor::Filler::ZEROS);

      for (int pt(0); pt < num_qps_; ++pt) {
        // refValues(numPlaneNodes, numQPs) = shape function
        // refArea(numCells, numQPs) = |Jacobian|*weight
        f_plus(0) += cohesive_traction_(cell, pt, 0) *
                     ref_values_(bottom_node, pt) * ref_area_(cell, pt);
        f_plus(1) += cohesive_traction_(cell, pt, 1) *
                     ref_values_(bottom_node, pt) * ref_area_(cell, pt);
        f_plus(2) += cohesive_traction_(cell, pt, 2) *
                     ref_values_(bottom_node, pt) * ref_area_(cell, pt);

      }  // end of pt loop

      force_(cell, bottom_node, 0) = -f_plus(0);
      force_(cell, bottom_node, 1) = -f_plus(1);
      force_(cell, bottom_node, 2) = -f_plus(2);

      force_(cell, top_node, 0) = f_plus(0);
      force_(cell, top_node, 1) = f_plus(1);
      force_(cell, top_node, 2) = f_plus(2);

    }  // end of planeNode loop
  }    // end of cell loop
}
//**********************************************************************
}  // namespace LCM
