//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATICTEMPERATURERESID_HPP
#define AERAS_XZHYDROSTATICTEMPERATURERESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace Aeras {
/** \brief XZHydrostatic Temperature equation Residual for atmospheric modeling

    This evaluator computes the residual of the XZHydrostatic Temperature 
    equation for atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_TemperatureResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_TemperatureResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>         wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,Node,Level>     temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim> temperatureGrad;
  PHX::MDField<ScalarT,Cell,Node,Level>     temperatureDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>     temperatureSrc;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> velocity;
  PHX::MDField<ScalarT,Cell,Node,Level>     omega;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>     etadotdT;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level> Residual;

  ScalarT Re; // Reynolds number (demo on how to get info from input file)

  ScalarT Cp;
  ScalarT Prandtl;
  ScalarT viscosity;

  const int numNodes;
  const int numQPs;
  const int numDims;
  const int numLevels;

  bool obtainLaplaceOp;
  bool pureAdvection;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_TemperatureResid_Tag{};
  struct XZHydrostatic_TemperatureResid_pureAdvection_Tag{};
  struct XZHydrostatic_TemperatureResid_Laplace_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_TemperatureResid_Tag> XZHydrostatic_TemperatureResid_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_TemperatureResid_Tag> XZHydrostatic_TemperatureResid_pureAdvection_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_TemperatureResid_Tag> XZHydrostatic_TemperatureResid_Laplace_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_TemperatureResid_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_TemperatureResid_pureAdvection_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_TemperatureResid_Laplace_Tag& tag, const int& i) const;

#endif
};
}

#endif
