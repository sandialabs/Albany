//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Intrepid_MiniTensor.h"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Sacado_MathFunctions.hpp"

namespace LCM {

//
//
//
template<typename EvalT, typename Traits>
SurfaceBasis<EvalT, Traits>::SurfaceBasis(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    need_current_basis_(false),
    reference_coords_(
        p.get<std::string>("Reference Coordinates Name"),
        dl->vertices_vector),
    cubature_(p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepid_basis_(
        p.get<Teuchos::RCP<Intrepid::Basis<RealType,
        Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    ref_basis_(p.get<std::string>("Reference Basis Name"), dl->qp_tensor),
    ref_area_(p.get<std::string>("Reference Area Name"), dl->qp_scalar),
    ref_dual_basis_(
        p.get<std::string>("Reference Dual Basis Name"),
        dl->qp_tensor),
    ref_normal_(p.get<std::string>("Reference Normal Name"), dl->qp_vector)
{
  this->addDependentField(reference_coords_);
  this->addEvaluatedField(ref_basis_);
  this->addEvaluatedField(ref_area_);
  this->addEvaluatedField(ref_dual_basis_);
  this->addEvaluatedField(ref_normal_);

  // if current coordinates are being passed in,
  // compute and return the current basis
  // needed for the localization element, but not uncoupled transport
  if (p.isType<std::string>("Current Coordinates Name")) {
    need_current_basis_ = true;

    // grab the current coords
    PHX::MDField<ScalarT, Cell, Vertex, Dim> tmp(
        p.get<std::string>("Current Coordinates Name"),
        dl->node_vector);
    current_coords_ = tmp;

    // set up the current basis
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> tmp2(
        p.get<std::string>("Current Basis Name"),
        dl->qp_tensor);
    current_basis_ = tmp2;

    this->addDependentField(current_coords_);
    this->addEvaluatedField(current_basis_);
  }

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);

  int containerSize = dims[0];
  num_nodes_ = dims[1];
  num_surf_nodes_ = num_nodes_ / 2;

  num_qps_ = cubature_->getNumPoints();
  num_surf_dims_ = cubature_->getDimension();
  num_dims_ = num_surf_dims_ + 1;

#ifdef ALBANY_VERBOSE
  std::cout << "in Surface Basis" << '\n';
  std::cout << " numPlaneNodes: " << num_surf_nodes_ << '\n';
  std::cout << " numPlaneDims: " << num_surf_dims_ << '\n';
  std::cout << " numQPs: " << num_qps_ << '\n';
  std::cout << " cubature->getNumPoints(): ";
  std::cout << cubature_->getNumPoints() << '\n';
  std::cout << " cubature->getDimension(): ";
  std::cout << cubature_->getDimension() << '\n';
#endif

  // Allocate Temporary FieldContainers
  ref_values_.resize(num_surf_nodes_, num_qps_);
  ref_grads_.resize(num_surf_nodes_, num_qps_, num_surf_dims_);
  ref_points_.resize(num_qps_, num_surf_dims_);
  ref_weights_.resize(num_qps_);

  // temp space for midplane coords
  ref_midplane_coords_.resize(containerSize, num_surf_nodes_, num_dims_);
  current_midplane_coords_.resize(containerSize, num_surf_nodes_, num_dims_);

  // Pre-Calculate reference element quantitites
  cubature_->getCubature(ref_points_, ref_weights_);
  intrepid_basis_->getValues(
      ref_values_, ref_points_, Intrepid::OPERATOR_VALUE);
  intrepid_basis_->getValues(ref_grads_, ref_points_, Intrepid::OPERATOR_GRAD);

  this->setName("SurfaceBasis" + PHX::typeAsString<EvalT>());
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(reference_coords_, fm);
  this->utils.setFieldData(ref_area_, fm);
  this->utils.setFieldData(ref_dual_basis_, fm);
  this->utils.setFieldData(ref_normal_, fm);
  this->utils.setFieldData(ref_basis_, fm);
  if (need_current_basis_) {
    this->utils.setFieldData(current_coords_, fm);
    this->utils.setFieldData(current_basis_, fm);
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int cell(0); cell < workset.numCells; ++cell) {
    // for the reference geometry
    // compute the mid-plane coordinates
    computeReferenceMidplaneCoords(reference_coords_, ref_midplane_coords_);

    // compute basis vectors
    computeReferenceBaseVectors(ref_midplane_coords_, ref_basis_);

    // compute the dual
    computeDualBaseVectors(
        ref_midplane_coords_,
        ref_basis_,
        ref_normal_,
        ref_dual_basis_);

    // compute the Jacobian
    computeJacobian(ref_basis_, ref_dual_basis_, ref_area_);

    if (need_current_basis_) {
      // for the current configuration
      // compute the mid-plane coordinates
      computeCurrentMidplaneCoords(current_coords_, current_midplane_coords_);

      // compute base vectors
      computeCurrentBaseVectors(current_midplane_coords_, current_basis_);
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::computeReferenceMidplaneCoords(
    PHX::MDField<MeshScalarT, Cell, Vertex, Dim> coords,
    MFC & midplaneCoords)
{
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
    // compute the mid-plane coordinates
    for (int node(0); node < num_surf_nodes_; ++node) {
      int topNode = node + num_surf_nodes_;
      for (int dim(0); dim < num_dims_; ++dim) {
        midplaneCoords(cell, node, dim) = 0.5
            * (coords(cell, node, dim) + coords(cell, topNode, dim));
      }
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::computeCurrentMidplaneCoords(
    PHX::MDField<ScalarT, Cell, Vertex, Dim> coords,
    SFC & midplaneCoords)
{
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
    // compute the mid-plane coordinates
    for (int node(0); node < num_surf_nodes_; ++node) {
      int topNode = node + num_surf_nodes_;
      for (int dim(0); dim < num_dims_; ++dim) {
        midplaneCoords(cell, node, dim) = 0.5
            * (coords(cell, node, dim) + coords(cell, topNode, dim));
      }
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::
computeReferenceBaseVectors(const MFC & midplaneCoords,
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis)
{
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
    // get the midplane coordinates
    std::vector<Intrepid::Vector<MeshScalarT> > midplaneNodes(num_surf_nodes_);
    for (int node(0); node < num_surf_nodes_; ++node)
      midplaneNodes[node] = Intrepid::Vector<MeshScalarT>(
          3,
          midplaneCoords,
          cell,
          node,
          0);

    Intrepid::Vector<MeshScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0);
    //compute the base vectors
    for (int pt(0); pt < num_qps_; ++pt) {
      g_0.clear();
      g_1.clear();
      g_2.clear();
      for (int node(0); node < num_surf_nodes_; ++node) {
        g_0 += ref_grads_(node, pt, 0) * midplaneNodes[node];
        g_1 += ref_grads_(node, pt, 1) * midplaneNodes[node];
      }
      g_2 = cross(g_0, g_1) / norm(cross(g_0, g_1));

      basis(cell, pt, 0, 0) = g_0(0);
      basis(cell, pt, 0, 1) = g_0(1);
      basis(cell, pt, 0, 2) = g_0(2);
      basis(cell, pt, 1, 0) = g_1(0);
      basis(cell, pt, 1, 1) = g_1(1);
      basis(cell, pt, 1, 2) = g_1(2);
      basis(cell, pt, 2, 0) = g_2(0);
      basis(cell, pt, 2, 1) = g_2(1);
      basis(cell, pt, 2, 2) = g_2(2);
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::
computeCurrentBaseVectors(const SFC & midplaneCoords,
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> basis)
{
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
    // get the midplane coordinates
    std::vector<Intrepid::Vector<ScalarT> > midplaneNodes(num_surf_nodes_);
    for (int node(0); node < num_surf_nodes_; ++node)
      midplaneNodes[node] = Intrepid::Vector<ScalarT>(
          3,
          midplaneCoords,
          cell,
          node,
          0);

    Intrepid::Vector<ScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0);
    //compute the base vectors
    for (int pt(0); pt < num_qps_; ++pt) {
      g_0.clear();
      g_1.clear();
      g_2.clear();
      for (int node(0); node < num_surf_nodes_; ++node) {
        g_0 += ref_grads_(node, pt, 0) * midplaneNodes[node];
        g_1 += ref_grads_(node, pt, 1) * midplaneNodes[node];
      }
      g_2 = cross(g_0, g_1) / norm(cross(g_0, g_1));

      basis(cell, pt, 0, 0) = g_0(0);
      basis(cell, pt, 0, 1) = g_0(1);
      basis(cell, pt, 0, 2) = g_0(2);
      basis(cell, pt, 1, 0) = g_1(0);
      basis(cell, pt, 1, 1) = g_1(1);
      basis(cell, pt, 1, 2) = g_1(2);
      basis(cell, pt, 2, 0) = g_2(0);
      basis(cell, pt, 2, 1) = g_2(1);
      basis(cell, pt, 2, 2) = g_2(2);
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::computeDualBaseVectors(
    const MFC & midplaneCoords,
    const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis,
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> normal,
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> dualBasis)
{
  int worksetSize = midplaneCoords.dimension(0);

  Intrepid::Vector<MeshScalarT>
  g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0);

  Intrepid::Vector<MeshScalarT>
  g0(0, 0, 0), g1(0, 0, 0), g2(0, 0, 0);

  for (int cell(0); cell < worksetSize; ++cell) {
    for (int pt(0); pt < num_qps_; ++pt) {
      g_0 = Intrepid::Vector<MeshScalarT>(3, basis, cell, pt, 0, 0);
      g_1 = Intrepid::Vector<MeshScalarT>(3, basis, cell, pt, 1, 0);
      g_2 = Intrepid::Vector<MeshScalarT>(3, basis, cell, pt, 2, 0);

      normal(cell, pt, 0) = g_2(0);
      normal(cell, pt, 1) = g_2(1);
      normal(cell, pt, 2) = g_2(2);

      g0 = cross(g_1, g_2) / dot(g_0, cross(g_1, g_2));
      g1 = cross(g_0, g_2) / dot(g_1, cross(g_0, g_2));
      g2 = cross(g_0, g_1) / dot(g_2, cross(g_0, g_1));

      dualBasis(cell, pt, 0, 0) = g0(0);
      dualBasis(cell, pt, 0, 1) = g0(1);
      dualBasis(cell, pt, 0, 2) = g0(2);
      dualBasis(cell, pt, 1, 0) = g1(0);
      dualBasis(cell, pt, 1, 1) = g1(1);
      dualBasis(cell, pt, 1, 2) = g1(2);
      dualBasis(cell, pt, 2, 0) = g2(0);
      dualBasis(cell, pt, 2, 1) = g2(1);
      dualBasis(cell, pt, 2, 2) = g2(2);
    }
  }
}

//
//
//
template<typename EvalT, typename Traits>
void SurfaceBasis<EvalT, Traits>::computeJacobian(
    const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis,
    const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> dualBasis,
    PHX::MDField<MeshScalarT, Cell, QuadPoint> area)
{
  const int worksetSize = basis.dimension(0);

  for (int cell(0); cell < worksetSize; ++cell) {
    for (int pt(0); pt < num_qps_; ++pt) {
      Intrepid::Tensor<MeshScalarT> dPhiInv(3, dualBasis, cell, pt, 0, 0);
      Intrepid::Tensor<MeshScalarT> dPhi(3, basis, cell, pt, 0, 0);
      Intrepid::Vector<MeshScalarT> G_2(3, basis, cell, pt, 2, 0);
      MeshScalarT j0 = Intrepid::det(dPhi);
      MeshScalarT jacobian = j0
          * std::sqrt(
              Intrepid::dot(
                  Intrepid::dot(G_2, Intrepid::transpose(dPhiInv) * dPhiInv),
                  G_2));
      area(cell, pt) = jacobian * ref_weights_(pt);
    }
  }

}

}//namespace LCM
