//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_MAP_THICKNESS_HPP
#define LANDICE_MAP_THICKNESS_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{


template<typename EvalT, typename Traits>
class MapThickness: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  
  MapThickness (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  // Input:
  PHX::MDField<const MeshScalarT> H_in;
  PHX::MDField<const MeshScalarT> H_min;
  PHX::MDField<const MeshScalarT> H_max;
  PHX::MDField<const MeshScalarT> H_obs;
  PHX::MDField<const MeshScalarT> bed;

  // Output:
  PHX::MDField<MeshScalarT> H_out;
};

} // Namespace LandIce

#endif // LANDICE_SIMPLE_OPERATION_HPP
