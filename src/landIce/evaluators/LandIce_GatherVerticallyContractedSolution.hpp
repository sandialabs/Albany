//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_GATHER_VERTICALLY_CONTRACTED_SOLUTION_HPP
#define LANDICE_GATHER_VERTICALLY_CONTRACTED_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "PHAL_AlbanyTraits.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class GatherVerticallyContractedSolution
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  GatherVerticallyContractedSolution (const Teuchos::ParameterList& p,
                                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  enum ContractionOperator {VerticalAverage, VerticalSum};

  Teuchos::ArrayRCP<const ST> x_constView;

protected:

  using ScalarT     = typename EvalT::ScalarT;
  using ExecSpace   = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

  using ref_t       = typename PHAL::Ref<ScalarT>::type;

  ref_t get_ref (const int iside, const int node, const int cmp) const {
    if (isVector) {
      return contractedSol(iside,node,cmp);
    } else {
      return contractedSol(iside,node);
    }
  }

  void computeQuadWeights(const Teuchos::ArrayRCP<double>& layers_ratio);
  void computeSideDOFOffsets (const Albany::DOFManager& dof_mgr);

  // Output:
  PHX::MDField<ScalarT>  contractedSol;

  bool isVector;

  int offset;

  int vecDim;
  int numNodes;

  std::string meshPart;

  ContractionOperator op;

  Albany::DualView<double*>   quadWeights;
  Albany::DualView<int*>      side_node_count;

  int numLayers;

  bool quad_weights_computed = false;

public:
  struct SolAccessor {
    bool isVector;
    Kokkos::DynRankView<ScalarT,PHX::Device> d_sol;

    KOKKOS_INLINE_FUNCTION
    ref_t get_ref (const int iside, const int node, const int cmp) const {
      if (isVector) {
        return d_sol(iside,node,cmp);
      } else {
        return d_sol(iside,node);
      }
    }
  };

  SolAccessor device_sol;
};


} // namespace LandIce

#endif // LANDICE_GATHER_VERTICALLY_CONTRACTED_SOLUTION_HPP
