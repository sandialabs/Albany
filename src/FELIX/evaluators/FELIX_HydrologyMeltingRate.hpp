//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_MELTING_RATE_HPP
#define FELIX_HYDROLOGY_MELTING_RATE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, bool IsStokes>
class HydrologyMeltingRate;

// Partial specialization for Hydrology only problem
template<typename EvalT, typename Traits>
class HydrologyMeltingRate<EvalT,Traits,false> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologyMeltingRate (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>  u_b;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>  beta;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>  G;

  // Output:
  PHX::MDField<ParamScalarT,Cell,QuadPoint>  m;

  int numQPs;

  double L;
};

// Partial specialization for StokesFO coupling
template<typename EvalT, typename Traits>
class HydrologyMeltingRate<EvalT,Traits,true> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  HydrologyMeltingRate (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>       u_b;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>       beta;
  PHX::MDField<const ParamScalarT,Cell,Side,QuadPoint>  G;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>       m;

  std::string       sideSetName;

  int numQPs;

  double L;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_MELTING_RATE_HPP
