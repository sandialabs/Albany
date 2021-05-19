//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_EFFECTIVE_PRESSURE_HPP
#define LANDICE_EFFECTIVE_PRESSURE_HPP 1

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/** \brief Effective Pressure Evaluator

    This evaluator computes the effective pressure N
    Note: if Surrogate=true, then we compute N as
             N = alpha*P_o, with 0<alpha<1.
          that is, a fraction of the overburden.
*/

template<typename EvalT, typename Traits, bool Surrogate>
class EffectivePressure : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  typedef typename std::conditional<Surrogate,ParamScalarT,ScalarT>::type  HydroScalarT;

  EffectivePressure (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const RealType>      P_o;
  PHX::MDField<const HydroScalarT>  P_w;

  // Output:
  PHX::MDField<HydroScalarT>  N;

  int numPts;
  unsigned int worksetSize;

  bool eval_on_side;
  std::string sideSetName; // Needed if OnSide=true

  PHX::MDField<const ScalarT,Dim> alphaParam;
  ScalarT printedAlpha;
  
  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct EffectivePressure_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, EffectivePressure_Tag> EffectivePressure_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const EffectivePressure_Tag& tag, const int& cell) const;
  
};

} // Namespace LandIce

#endif // LANDICE_EFFECTIVE_PRESSURE_HPP
