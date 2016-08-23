//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_ETADOTPI_HPP
#define AERAS_XZHYDROSTATIC_ETADOTPI_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Aeras_Eta.hpp"
#include "Kokkos_Vector.hpp"

namespace Aeras {
/** \brief Density for XZHydrostatic atmospheric model

    This evaluator computes the density for the XZHydrostatic model
    of atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_EtaDotPi : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      divpivelx;
  PHX::MDField<ScalarT,Cell,Node>            pdotP0;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      Pi;
  PHX::MDField<ScalarT,Cell,Node,Level>      Temperature;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  Velocity;

  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      etadotdT;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      etadot;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  etadotdVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      Pidot;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint,Level> > Tracer;
  //std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint,Level> > etadotdTracer;
  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint,Level> > dedotpiTracerde;

#else
  Kokkos::vector< PHX::MDField<ScalarT,Cell,QuadPoint,Level>, PHX::Device > Tracer;
  //Kokkos::vector< PHX::MDField<ScalarT,Cell,QuadPoint,Level>, PHX::Device > etadotdTracer;
  Kokkos::vector< PHX::MDField<ScalarT,Cell,QuadPoint,Level>, PHX::Device > dedotpiTracerde;

#endif

  const Teuchos::ArrayRCP<std::string> tracerNames;
  //const Teuchos::ArrayRCP<std::string> etadotdtracerNames;
  const Teuchos::ArrayRCP<std::string> dedotpitracerdeNames;

  const int numQPs;
  const int numDims;
  const int numLevels;
  const Eta<EvalT> &E;

  bool pureAdvection;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_EtaDotPi_Tag{};
  struct XZHydrostatic_EtaDotPi_pureAdvection_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_EtaDotPi_Tag> XZHydrostatic_EtaDotPi_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_EtaDotPi_pureAdvection_Tag> XZHydrostatic_EtaDotPi_pureAdvection_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_EtaDotPi_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_EtaDotPi_pureAdvection_Tag& tag, const int& i) const;

#endif
};
}

#endif
