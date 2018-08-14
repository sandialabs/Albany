//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

template <typename ScalarT>
inline ScalarT
Sqr(const ScalarT& num)
{
  return num * num;
}

namespace LCM {

//----------------------------------------------------------------------------
template <typename Traits>
IsoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::IsoMeshSizeField(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : MeshSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits>(dl),
      currentCoords(
          p.get<std::string>("Current Coordinates Name"),
          dl->node_vector),
      isoMeshSizeField(
          p.get<std::string>("IsoTropic MeshSizeField Name"),
          dl->qp_scalar),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis"))

{
  // Save the adaptation PL to pass back

  adapt_PL = p.get<Teuchos::ParameterList*>("Parameter List");

  // Set the value to disable adaptation initially

  adapt_PL->set<bool>("AdaptNow", false);

  this->addDependentField(currentCoords);

  this->addEvaluatedField(isoMeshSizeField);

  this->setName("IsoMeshSizeField<Residual>");

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::vector<PHX::DataLayout::size_type> dims2;
  dl->node_vector->dimensions(dims2);
  numNodes = dims2[1];
}

//----------------------------------------------------------------------------
template <typename Traits>
void
IsoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(isoMeshSizeField, fm);
  this->utils.setFieldData(currentCoords, fm);

  // Allocate Temporary Views
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", numNodes, numQPs, numDims);
  refPoints =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);
  dxdxi      = Kokkos::createDynRankView(
      isoMeshSizeField.get_view(), "XXX", numDims, numDims);
  dEDdxi =
      Kokkos::createDynRankView(isoMeshSizeField.get_view(), "XXX", numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(
      grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);
}

//----------------------------------------------------------------------------
template <typename Traits>
void
IsoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Compute IsoMeshSizeField - element width is dx/dxi (sum_nodes_i x[i] *
  // dphi[i]/dxi[j])
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int dim = 0; dim < numDims; ++dim) {
        for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
          dxdxi(dim, ref_dim) = 0.0;
        }
      }

      for (int dim = 0; dim < numDims; ++dim) {
        for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
          for (int node = 0; node < numNodes; ++node) {
            // Calculate the element width, in each of the x, y, and z
            // dimensions grad_at_cub_points(node, qp, dim) - node corresponds
            // to the element node (local)
            //                                   - qp is the integration point
            //                                   - ref_dim is xi, etc, zeta
            //                                   dimension
            dxdxi(dim, ref_dim) += currentCoords(cell, node, dim) *
                                   grad_at_cub_points(node, qp, ref_dim);
          }
        }
      }

      for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
        dEDdxi(ref_dim) = 0.0;
      }

      // Calculate Euclidean distance of the element in each of the master
      // directions
      for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
        for (int dim = 0; dim < numDims; ++dim) {
          dEDdxi(ref_dim) += Sqr(dxdxi(dim, ref_dim));
        }
        dEDdxi(ref_dim) = std::sqrt(dEDdxi(ref_dim));
      }

      isoMeshSizeField(cell, qp) = 0.0;

      for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
        isoMeshSizeField(cell, qp) += dEDdxi(ref_dim);
      }

      isoMeshSizeField(cell, qp) /= double(numDims);
    }
  }
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <typename Traits>
AnisoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::AnisoMeshSizeField(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : MeshSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits>(dl),
      currentCoords(
          p.get<std::string>("Current Coordinates Name"),
          dl->node_vector),
      anisoMeshSizeField(
          p.get<std::string>("AnisoTropic MeshSizeField Name"),
          dl->qp_scalar),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis"))

{
  // Save the adaptation PL to pass back

  adapt_PL = p.get<Teuchos::ParameterList*>("Parameter List");

  // Set the value to disable adaptation initially

  adapt_PL->set<bool>("AdaptNow", false);

  this->addDependentField(currentCoords);

  this->addEvaluatedField(anisoMeshSizeField);

  this->setName("AnisoMeshSizeField<Residual>");

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::vector<PHX::DataLayout::size_type> dims2;
  dl->node_vector->dimensions(dims2);
  numNodes = dims2[1];
}

//----------------------------------------------------------------------------
template <typename Traits>
void
AnisoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(anisoMeshSizeField, fm);
  this->utils.setFieldData(currentCoords, fm);

  // Allocate Temporary Views
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", numNodes, numQPs, numDims);
  refPoints =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);
  dxdxi      = Kokkos::createDynRankView(
      anisoMeshSizeField.get_view(), "XXX", numDims, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(
      grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);
}

//----------------------------------------------------------------------------
template <typename Traits>
void
AnisoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Compute IsoMeshSizeField tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int dim = 0; dim < numDims; ++dim) {
        for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
          dxdxi(dim, ref_dim) = 0.0;
        }
      }

      for (int dim = 0; dim < numDims; ++dim) {
        for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
          for (int node = 0; node < numNodes; ++node) {
            // Calculate the element width, in each of the x, y, and z
            // dimensions grad_at_cub_points(node, qp, dim) - node corresponds
            // to the element node (local)
            //                                   - qp is the integration point
            //                                   - ref_dim is xi, etc, zeta
            //                                   dimension
            dxdxi(dim, ref_dim) += currentCoords(cell, node, dim) *
                                   grad_at_cub_points(node, qp, ref_dim);
          }
        }
      }

      for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
        anisoMeshSizeField(cell, qp, ref_dim) = 0.0;
      }

      // Calculate Euclidean distance of the element in each of the master
      // directions
      for (int ref_dim = 0; ref_dim < numDims; ++ref_dim) {  // xi, eta, zeta
        for (int dim = 0; dim < numDims; ++dim) {
          anisoMeshSizeField(cell, qp, ref_dim) += Sqr(dxdxi(dim, ref_dim));
        }
        anisoMeshSizeField(cell, qp, ref_dim) =
            std::sqrt(anisoMeshSizeField(cell, qp, ref_dim));
      }
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "NOTE: Please remember that the Anisotropic size field is defined as a "
      "vector in xi, etc, zeta space not x, y, z!!!");
}
//----------------------------------------------------------------------------
}  // namespace LCM
