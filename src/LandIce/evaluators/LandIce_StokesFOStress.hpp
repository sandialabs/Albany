//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_STRESS_HPP
#define LANDICE_STOKES_FO_STRESS_HPP

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, typename SurfHeightST>
class StokesFOStress : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  StokesFOStress(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData /* d */,
                             PHX::FieldManager<Traits>& /* vm */) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const SurfHeightST,Cell,QuadPoint>         surfaceHeight;

  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim,Dim>   Ugrad;

  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim>       U;

  PHX::MDField<const ScalarT,Cell,QuadPoint>              muLandIce;

  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim>      coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>     Stress;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDimFO;
  Teuchos::ParameterList* stereographicMapList;
  bool useStereographicMap;
  double rho_g;
};

} // namespace LandIce

#endif // LANDICE_STOKES_FO_STRESS_HPP
