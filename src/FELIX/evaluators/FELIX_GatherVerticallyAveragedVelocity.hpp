//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_GATHERVERTICALLYAVERAGEDVELOCITY_HPP
#define FELIX_GATHERVERTICALLYAVERAGEDVELOCITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "PHAL_AlbanyTraits.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class GatherVerticallyAveragedVelocityBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  GatherVerticallyAveragedVelocityBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~GatherVerticallyAveragedVelocityBase(){};

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

protected:


  typedef typename EvalT::ScalarT ScalarT;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim>  averagedVel;

  std::size_t vecDim;
  std::size_t vecDimFO;
  std::size_t numNodes;
};


template<typename EvalT, typename Traits> class GatherVerticallyAveragedVelocity;



template<typename Traits>
class GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Residual,Traits>
    : public GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

  GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Jacobian,Traits>
    : public GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Tangent,Traits>
    : public GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};



}

#endif
