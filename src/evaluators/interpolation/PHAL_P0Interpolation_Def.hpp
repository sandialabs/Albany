//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_P0Interpolation.hpp"

#include "Albany_DiscretizationUtils.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
P0InterpolationBase<EvalT, Traits, ScalarT>::
P0InterpolationBase (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl)
{
  using FL  = Albany::FieldLocation;
  using FRT = Albany::FieldRankType;

  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  // Check what interpolation type we need to do
  std::string interpType = p.get<std::string>("Interpolation Type");
  TEUCHOS_TEST_FOR_EXCEPTION ((interpType != "Value At Cell Barycenter") && (interpType != "Cell Average"),
      std::runtime_error, "Interpolation Type can either be \"Value At Cell Barycenter\" or \"Cell Average\"");

  itype = (interpType == "Value At Cell Barycenter") ? ValueAtCellBarycenter : CellAverage;

  // Barycenter value needs Intrepid2 Basis, while Average needs cell w_measure
  if(itype == ValueAtCellBarycenter) {
    intrepidBasis = p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis");
    TEUCHOS_TEST_FOR_EXCEPTION (intrepidBasis.is_null(), std::runtime_error,
        "intrepidBasis needs to be nonnull when Interpolation Type is \"Value At Cell Barycenter\"");
  } else {
    // For CellAverage, we need to divide by cell measure
    w_measure = decltype(w_measure)(p.get<std::string>("Weighted Measure Name"), dl->qp_scalar);
    this->addDependentField(w_measure);
  }

  // Determine input and output layouts, and get dimensions
  Teuchos::RCP<PHX::DataLayout> point_layout, p0_layout;

  loc  = p.get<FL>("Field Location");
  rank = p.get<FRT>("Field Rank Type");

  TEUCHOS_TEST_FOR_EXCEPTION (loc!=FL::Node && loc!=FL::QuadPoint, std::logic_error,
      "Error! P0 interpolation evaluator requires an input Node or QuadPoint field.\n");

  point_layout = get_field_layout(rank,loc,dl);
  p0_layout = get_field_layout(rank,FL::Cell,dl);

  if (rank!=FRT::Scalar) {
    dim0 = p0_layout->dimension(1);
    if (rank==FRT::Tensor) {
      dim1 = p0_layout->dimension(2);
    }
  }
  numQPs   = dl->qp_scalar->dimension(1);
  numNodes = dl->node_scalar->dimension(1);

  // Create input/output fields
  field    = decltype(field)(p.get<std::string> ("Field Name"), point_layout);
  this->addDependentField (field);

  if (itype==CellAverage) {
    field_avg = decltype(field_avg)(p.get<std::string> ("Field P0 Name"), p0_layout);
    this->addEvaluatedField (field_avg);
  } else {
    field_baryc = decltype(field_baryc)(p.get<std::string> ("Field P0 Name"), p0_layout);
    this->addEvaluatedField (field_baryc);
  }


  // Check if input is nodal or quad points
  if (loc==FL::Node) {
    // If itype is CellAverage, we need basis and measure too
    if (itype == CellAverage) {
      BF = decltype(BF) (p.get<std::string>("BF Name"), dl->node_qp_scalar);
      this->addDependentField(BF);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (itype==ValueAtCellBarycenter, std::logic_error,
        "Error! Barycenter interpolation requires a Node field as input, not a QuadPoint field.\n");
  }

  this->setName("P0Interpolation[" + p.get<std::string> ("Field Name") + "] "+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename ScalarT>
void P0InterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) {
    memoizer.enable_memoizer();
  }

  if(itype == ValueAtCellBarycenter) {
    basis_at_barycenter = Kokkos::DynRankView<RealType, PHX::Device>("basis_at_barycenter", numNodes, 1);
    auto cellTopology = intrepidBasis->getBaseCellTopology();
    Kokkos::DynRankView<RealType, PHX::Device> refPoint("refPoint", 1, cellTopology.getDimension());
    Kokkos::DynRankView<RealType, PHX::Device> refWeight("refWeights", 1);

    // Pre-Calculate reference element quantities
    Intrepid2::DefaultCubatureFactory cubFactory;
    auto onePointCubature = cubFactory.create<PHX::Device, RealType, RealType>(cellTopology, 0);
    onePointCubature->getCubature(refPoint, refWeight);
    intrepidBasis->getValues(basis_at_barycenter, refPoint, Intrepid2::OPERATOR_VALUE);
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void P0InterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluate_on_side(workset);
  } else {
    evaluate_on_cell(workset.numCells);
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void P0InterpolationBase<EvalT, Traits, ScalarT>::evaluate_on_cell (const int numCells)
{
  using FRT = Albany::FieldRankType;
  if (itype == CellAverage) {
    switch (rank) {
      case FRT::Scalar:
        Kokkos::parallel_for(Cell_Average_Scalar_Field_Policy(0, numCells), *this);
        break;
      case FRT::Vector:
      case FRT::Gradient:
        Kokkos::parallel_for(Cell_Average_Vector_Field_Policy(0, numCells), *this);
        break;
      case FRT::Tensor:
        Kokkos::parallel_for(Cell_Average_Tensor_Field_Policy(0, numCells), *this);
        break;
    }
  } else {
    switch (rank) {
      case FRT::Scalar:
        Kokkos::parallel_for(Cell_Barycenter_Scalar_Field_Policy(0, numCells), *this);
        break;
      case FRT::Vector:
      case FRT::Gradient:
        Kokkos::parallel_for(Cell_Barycenter_Vector_Field_Policy(0, numCells), *this);
        break;
      case FRT::Tensor:
        Kokkos::parallel_for(Cell_Barycenter_Tensor_Field_Policy(0, numCells), *this);
        break;
    }
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void P0InterpolationBase<EvalT, Traits, ScalarT>::evaluate_on_side (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  const auto& sideSet = workset.sideSets->at(sideSetName);

  evaluate_on_cell(sideSet.size());
}

//**********************************************************************
// Kokkos operators
//**********************************************************************

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Average_Scalar_Field_Tag&, const int& cell) const{
  using FL = Albany::FieldLocation;

  ScalarT field_qp_device;

  // Precompute cell measure
  MeshScalarT meas_device = 0.0;
  for (int qp(0); qp<numQPs; ++qp) {
    meas_device += w_measure(cell,qp);
  }

  // Zero out the output
  field_avg(cell) = 0;

  // Integrate over cell
  for (int qp(0); qp<numQPs; ++qp) {
    const auto w_meas_qp = w_measure(cell,qp);
    if (loc==FL::Node) {
      field_qp_device = 0;
      for (int node(0); node<numNodes; ++node) {
        field_qp_device += field(cell,node)*BF(cell,node,qp);
      }
    } else {
      field_qp_device = field(cell,qp);
    }
    field_avg(cell) += field_qp_device*w_meas_qp;
  }

  // Scale by cell measure
  field_avg(cell) /= meas_device;
}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Average_Vector_Field_Tag&, const int& cell) const{
  using FL = Albany::FieldLocation;

  ScalarT field_qp_device;

  // Precompute cell measure
  MeshScalarT meas_device = 0.0;
  for (int qp(0); qp<numQPs; ++qp) {
    meas_device += w_measure(cell,qp);
  }

  // Zero out the output
  for (int dim(0); dim<dim0; ++dim) {
    field_avg(cell,dim) = 0;
  }

  // Integrate over cell
  for (int qp(0); qp<numQPs; ++qp) {
    auto w_meas_qp = w_measure(cell,qp);
    for (int dim(0); dim<dim0; ++dim) {
      if (loc==FL::Node) {
        field_qp_device = 0;
        for (int node(0); node<numNodes; ++node) {
          field_qp_device += field(cell,node,dim)*BF(cell,node,qp);
        }
      } else {
        field_qp_device = field(cell,qp,dim);
      }
      field_avg(cell,dim) += field_qp_device*w_meas_qp;
    }
  }

  // Scale by cell measure
  for (int dim(0); dim<dim0; ++dim) {
    field_avg(cell,dim) /= meas_device;
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Average_Tensor_Field_Tag&, const int& cell) const{
  using FL = Albany::FieldLocation;

  ScalarT field_qp_device;

  // Precompute cell measure
  MeshScalarT meas_device = 0.0;
  for (int qp(0); qp<numQPs; ++qp) {
    meas_device += w_measure(cell,qp);
  }

  // Zero out the output
  for (int idim(0); idim<dim0; ++idim) {
    for (int jdim(0); jdim<dim1; ++jdim) {
      field_avg(cell,idim,jdim) = 0;
  }}

  // Integrate over cell
  for (int qp(0); qp<numQPs; ++qp) {
    auto w_meas_qp = w_measure(cell,qp);
    for (int idim(0); idim<dim0; ++idim) {
      for (int jdim(0); jdim<dim1; ++jdim) {
        if (loc==FL::Node) {
          field_qp_device = 0;
          for (int node(0); node<numNodes; ++node) {
            field_qp_device += field(cell,node,idim,jdim)*BF(cell,node,qp);
          }
        } else {
          field_qp_device = field(cell,qp,idim,jdim);
        }
        field_avg(cell,idim,jdim) += field_qp_device*w_meas_qp;
    }}
  }

  // Scale by cell measure
  for (int idim(0); idim<dim0; ++idim) {
    for (int jdim(0); jdim<dim1; ++jdim) {
      field_avg(cell,idim,jdim) /= meas_device;
  }}
}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Barycenter_Scalar_Field_Tag&, const int& cell) const{
  field_baryc(cell) = 0;
  for (int node = 0; node<numNodes; ++node)
    field_baryc(cell) += field(cell,node)*basis_at_barycenter(node,0);
}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Barycenter_Vector_Field_Tag&, const int& cell) const{
  for (int dim = 0; dim<dim0; ++dim) {
    field_baryc(cell,dim) = 0;
  }
  for (int node = 0; node<numNodes; ++node) {
    const auto bab = basis_at_barycenter(node,0);
    for (int dim = 0; dim<dim0; ++dim) {
      field_baryc(cell,dim) += field(cell,node,dim)*bab;
  }}
}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void P0InterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Barycenter_Tensor_Field_Tag&, const int& cell) const{
  for (int idim = 0; idim<dim0; ++idim) {
    for (int jdim = 0; jdim<dim1; ++jdim) {
      field_baryc(cell,idim,jdim) = 0;
  }}

  for (int node = 0; node<numNodes; ++node) {
    const auto bab = basis_at_barycenter(node,0);
    for (int idim = 0; idim<dim0; ++idim) {
      for (int jdim = 0; jdim<dim1; ++jdim) {
        field_baryc(cell,idim,jdim) += field(cell,node,idim,jdim)*bab;
  }}}
}

} // Namespace PHAL
