//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LAMESTRESS_HPP
#define LAMESTRESS_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "lame/LameUtils.hpp"

namespace LCM {
/** \brief Evaluates stress using the Library for Advanced Materials for
 * Engineering (LAME).
 */

// Base class
// (1) Implements constructor and postRegistrationSetup for all inherited
// and specialized cases below. Implements a dummy evaluateFields that
// just throw's a Not Implemented
// (2) Implements private functions with calls to Lame with doubles.
//
template <typename EvalT, typename Traits>
class LameStressBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  LameStressBase(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  virtual void
  evaluateFields(typename Traits::EvalData d);

 protected:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Protected function for stress calc, only for RealType
  void
  calcStressRealType(
      PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldRef,
      PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& defGradFieldRef,
      typename Traits::EvalData                          workset,
      Teuchos::RCP<LameMatParams>&                       matp);

  // Allocate material parameter arrays -- always doubles
  void
  setMatP(Teuchos::RCP<LameMatParams>& matp, typename Traits::EvalData workset);

  // Free material pointer arrays -- always doubles
  void
  freeMatP(Teuchos::RCP<LameMatParams>& matp);

  // Input:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGradField;

  std::string                   defGradName, stressName;
  unsigned int                  numQPs;
  unsigned int                  numDims;
  Teuchos::RCP<PHX::DataLayout> tensor_dl;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stressField;

  // The LAME material model
  Teuchos::RCP<LameMaterial> lameMaterialModel;

  // The LAME material model name
  std::string lameMaterialModelName;

  // Vector of the state variable names for the LAME material model
  std::vector<std::string> lameMaterialModelStateVariableNames;

  // Vector of the fields corresponding to the LAME material model state
  // variables
  std::vector<PHX::MDField<ScalarT, Cell, QuadPoint>>
      lameMaterialModelStateVariableFields;

  // Work space
  PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim> stressFieldRealType,
      defGradFieldRealType;
};

// Inherted classes
template <typename EvalT, typename Traits>
class LameStress;

// For all cases except those specialized below, just fall through to base
// class. The base class throws "Not Implemented" for evaluate fields.
template <typename EvalT, typename Traits>
class LameStress : public LameStressBase<EvalT, Traits>
{
 public:
  LameStress(Teuchos::ParameterList& p) : LameStressBase<EvalT, Traits>(p){};
};

// Template Specialization: Residual Eval calls Lame with doubles.
template <typename Traits>
class LameStress<PHAL::AlbanyTraits::Residual, Traits>
    : public LameStressBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  LameStress(Teuchos::ParameterList& p)
      : LameStressBase<PHAL::AlbanyTraits::Residual, Traits>(p){};
  void
  evaluateFields(typename Traits::EvalData d);
};

// Template Specialization: Jacobian Eval does finite difference of Lame with
// doubles.
template <typename Traits>
class LameStress<PHAL::AlbanyTraits::Jacobian, Traits>
    : public LameStressBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  LameStress(Teuchos::ParameterList& p)
      : LameStressBase<PHAL::AlbanyTraits::Jacobian, Traits>(p){};
  void
  evaluateFields(typename Traits::EvalData d);
};

// Template Specialization: Tangent Eval does finite difference of Lame with
// doubles.
template <typename Traits>
class LameStress<PHAL::AlbanyTraits::Tangent, Traits>
    : public LameStressBase<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  LameStress(Teuchos::ParameterList& p)
      : LameStressBase<PHAL::AlbanyTraits::Tangent, Traits>(p){};
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
