//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  SurfaceScalarResidual<EvalT, Traits>::
  SurfaceScalarResidual(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness_      (p.get<double>("thickness")),
    cubature_       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepid_basis_ (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    ref_dual_basis_ (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    ref_normal_     (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    ref_area_       (p.get<std::string>("Reference Area Name"),dl->qp_scalar),
    residual_       (p.get<std::string>("Surface Scalar Residual Name"),dl->node_vector)
  {
    this->addDependentField(ref_dual_basis_);
    this->addDependentField(ref_normal_);    
    this->addDependentField(ref_area_);

    this->addEvaluatedField(residual_);

    this->setName("Surface Scalar Residual"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    num_nodes_ = dims[1];
    num_dims_ = dims[2];

    num_pts_ = cubature_->getNumPoints();

    num_plane_nodes_ = num_nodes_ / 2;
    num_plane_dims_ = num_dims_ - 1;

    // Allocate Temporary FieldContainers
    ref_values_.resize(num_plane_nodes, num_pts_);
    ref_grads_.resize(num_plane_nodes_, num_pts_, num_plane_dims_);
    ref_points_.resize(num_pts_, num_plane_dims_);
    ref_weights_.resize(num_pts_);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepid_basis_->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepid_basis_->getValues(ref_grads_, refPoints, Intrepid::OPERATOR_GRAD);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceScalarResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(ref_dual_basis_,fm);
    this->utils.setFieldData(ref_normal_,fm);
    this->utils.setFieldData(ref_area_,fm);
    this->utils.setFieldData(residual_,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceScalarResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
  }
  //----------------------------------------------------------------------------
}
