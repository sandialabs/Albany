//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_FLUXDIV_HPP
#define LANDICE_FLUXDIV_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_SacadoTypes.hpp"

#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits, typename ThicknessScalarT>
class FluxDiv : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  FluxDiv (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl_basal);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  using GradThicknessScalarT = typename Albany::StrongestScalarType<ThicknessScalarT,MeshScalarT>::type;

  // Input:
  PHX::MDField<const ScalarT>                                 field;
  PHX::MDField<const ScalarT,Side,QuadPoint,VecDim>           averaged_velocity;
  PHX::MDField<const ScalarT,Side,QuadPoint>                  div_averaged_velocity;
  PHX::MDField<const ThicknessScalarT,Side,QuadPoint>         thickness;
  PHX::MDField<const GradThicknessScalarT,Side,QuadPoint,Dim> grad_thickness;
  PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim>      side_tangents;

  // Output:
  PHX::MDField<ScalarT,Side,QuadPoint>                        flux_div;

  std::string sideSetName;
  unsigned int numSideQPs, numSideDims;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct FluxDiv_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, FluxDiv_Tag> FluxDiv_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const FluxDiv_Tag& tag, const int& sideSet_idx) const;

};

} // Namespace LandIce

#endif // LANDICE_FLUXDIV_HPP
