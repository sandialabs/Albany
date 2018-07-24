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

namespace LandIce
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits>
class FluxDiv : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  FluxDiv (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl_basal);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  // Input:
  PHX::MDField<const ScalarT> field;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint,VecDim>       averaged_velocity;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>              div_averaged_velocity;
  PHX::MDField<const ParamScalarT,Cell,Side,QuadPoint>         thickness;
  PHX::MDField<const ParamScalarT,Cell,Side,QuadPoint,Dim>     grad_thickness;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>  side_tangents;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>              flux_div;

  std::string sideSetName;
  int numSideQPs, numSideDims;
};

} // Namespace LandIce

#endif // LANDICE_FLUXDIV_HPP
