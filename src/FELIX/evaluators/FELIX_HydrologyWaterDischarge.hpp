//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
#define FELIX_HYDROLOGY_WATER_DISCHARGE_HPP 1

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

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
class HydrologyWaterDischarge;

// Partial specialization for Hydrology Only problem
template<typename EvalT, typename Traits, bool HasThicknessEqn>
class HydrologyWaterDischarge<EvalT,Traits,HasThicknessEqn,false> :
      public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<HasThicknessEqn,ScalarT,ParamScalarT>::type hScalarT;

  HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>  gradPhi;
  PHX::MDField<ScalarT,Cell,QuadPoint>      gradPhiNorm;
  PHX::MDField<hScalarT,Cell,QuadPoint>     h;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>  q;

  int numQPs;
  int numDim;

  double mu_w;
  double k_0;
  double alpha;
  double beta;

  bool needsGradPhiNorm;
};

// Partial specialization for StokesFO coupling
template<typename EvalT, typename Traits, bool HasThicknessEqn>
class HydrologyWaterDischarge<EvalT,Traits,HasThicknessEqn,true> :
      public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<HasThicknessEqn,ScalarT,ParamScalarT>::type hScalarT;

  HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>   gradPhi;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>       gradPhiNorm;
  PHX::MDField<hScalarT,Cell,Side,QuadPoint>      h;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>   q;

  int numQPs;
  int numDim;
  std::string   sideSetName;

  double mu_w;
  double k_0;
  double alpha;
  double beta;

  bool needsGradPhiNorm;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
