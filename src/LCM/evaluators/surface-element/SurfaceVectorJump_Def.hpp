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
SurfaceVectorJump<EvalT, Traits>::SurfaceVectorJump(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : cubature_(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepid_basis_(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      vector_(p.get<std::string>("Vector Name"), dl->node_vector),
      jump_(p.get<std::string>("Vector Jump Name"), dl->qp_vector)
{
  this->addDependentField(vector_);

  this->addEvaluatedField(jump_);

  this->setName("Surface Vector Jump" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_dims_     = dims[2];

  num_qps_ = cubature_->getNumPoints();

  num_plane_nodes_ = num_nodes_ / 2;
  num_plane_dims_  = num_dims_ - 1;

#ifdef ALBANY_VERBOSE
  std::cout << "in Surface Vector Jump" << '\n';
  std::cout << " num_plane_nodes_: " << num_plane_nodes_ << '\n';
  std::cout << " num_plane_dims_: " << num_plane_dims_ << '\n';
  std::cout << " num_qps_: " << num_qps_ << '\n';
  std::cout << " cubature_->getNumPoints(): ";
  std::cout << cubature_->getNumPoints() << '\n';
  std::cout << " cubature_->getDimension(): ";
  std::cout << cubature_->getDimension() << '\n';
#endif
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceVectorJump<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(vector_, fm);
  this->utils.setFieldData(jump_, fm);

  // Allocate Temporary Views
  ref_values_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_plane_nodes_, num_qps_);
  ref_grads_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_plane_nodes_, num_qps_, num_plane_dims_);
  ref_points_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_qps_, num_plane_dims_);
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
SurfaceVectorJump<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Vector<ScalarT> vecA(0, 0, 0), vecB(0, 0, 0), vecJump(0, 0, 0);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_qps_; ++pt) {
      vecA.fill(minitensor::Filler::ZEROS);
      vecB.fill(minitensor::Filler::ZEROS);
      for (int node = 0; node < num_plane_nodes_; ++node) {
        int topNode = node + num_plane_nodes_;
        vecA += minitensor::Vector<ScalarT>(
            ref_values_(node, pt) * vector_(cell, node, 0),
            ref_values_(node, pt) * vector_(cell, node, 1),
            ref_values_(node, pt) * vector_(cell, node, 2));
        vecB += minitensor::Vector<ScalarT>(
            ref_values_(node, pt) * vector_(cell, topNode, 0),
            ref_values_(node, pt) * vector_(cell, topNode, 1),
            ref_values_(node, pt) * vector_(cell, topNode, 2));
      }
      vecJump            = vecB - vecA;
      jump_(cell, pt, 0) = vecJump(0);
      jump_(cell, pt, 1) = vecJump(1);
      jump_(cell, pt, 2) = vecJump(2);
    }
  }
}

//**********************************************************************
}  // namespace LCM
