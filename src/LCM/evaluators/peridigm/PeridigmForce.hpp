//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PERIDIGMFORCE_HPP
#define PERIDIGMFORCE_HPP

#include "PeridigmManager.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include <Peridigm.hpp>
#include <Peridigm_AlbanyDiscretization.hpp>

namespace LCM {

/** \brief Evaluates nodal forces through a code coupling with the Peridigm
 * peridynamics code.
 */
template <typename EvalT, typename Traits>
class PeridigmForceBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  PeridigmForceBase(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dataLayout);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  Teuchos::ParameterList peridigmParams;

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  bool supportsTransient;

  // Input:
  RealType                                     density;
  PHX::MDField<RealType, Cell, Vertex>         sphereVolume;
  PHX::MDField<MeshScalarT, Cell, Vertex, Dim> referenceCoordinates;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>  currentCoordinates;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>  velocity;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>  acceleration;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>    force;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>    residual;
  std::vector<LCM::PeridigmManager::OutputField> outputFieldInfo;
  std::map<std::string, PHX::MDField<ScalarT>>   outputFields;

  unsigned int numQPs;
  unsigned int numDims;

  Teuchos::RCP<PeridigmNS::Peridigm> peridigm;
};

// Inherted classes
template <typename EvalT, typename Traits>
class PeridigmForce;

// For all cases except those specialized below, just fall through to base
// class. The base class throws "Not Implemented" for evaluate fields.
template <typename EvalT, typename Traits>
class PeridigmForce : public PeridigmForceBase<EvalT, Traits>
{
 public:
  PeridigmForce(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dataLayout)
      : PeridigmForceBase<EvalT, Traits>(p, dataLayout){};
};

// Template Specialization: Residual Evaluation (standard force evaluation)
template <typename Traits>
class PeridigmForce<PHAL::AlbanyTraits::Residual, Traits>
    : public PeridigmForceBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  PeridigmForce(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dataLayout)
      : PeridigmForceBase<PHAL::AlbanyTraits::Residual, Traits>(
            p,
            dataLayout){};
  void
  evaluateFields(typename Traits::EvalData d);
};

// Template Specialization: DistParamDeriv Evaluation
template <typename Traits>
class PeridigmForce<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public PeridigmForceBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  PeridigmForce(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dataLayout)
      : PeridigmForceBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(
            p,
            dataLayout){};
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
