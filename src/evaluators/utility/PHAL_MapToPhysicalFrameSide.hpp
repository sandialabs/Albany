//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP
#define PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Utilities.hpp"
#include "PHAL_Dimension.hpp"

namespace PHAL {
/** \brief Compute Side Quad points coordinates

    This evaluator computes the coordinates of the quad points
    on a side set, by interpolating the side vertices coordinates.

*/

template<typename EvalT, typename Traits>
class MapToPhysicalFrameSide : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  MapToPhysicalFrameSide (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Accessory variables
  Kokkos::DynRankView<RealType, PHX::Device> phi_at_cub_points[6];
  int numSideVertices[6];

  int numDim;
  int numSideQPs;
  std::string sideSetName;

  MDFieldMemoizer<Traits> memoizer;

  Albany::LocalSideSetInfo sideSet;

  // Input:
  //! Values at vertices
  PHX::MDField<const MeshScalarT,Side,Vertex,Dim> coords_side_vertices;

  // Output:
  //! Values at quadrature points
  PHX::MDField<MeshScalarT> coords_side_qp;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct MapToPhysicalFrameSide_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, MapToPhysicalFrameSide_Tag> MapToPhysicalFrameSide_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const MapToPhysicalFrameSide_Tag& tag, const int& sideSet_idx) const;

};

} // Namespace PHAL

#endif // PHAL_MAP_TO_PHYSICAL_FRAME_SIDE_HPP
