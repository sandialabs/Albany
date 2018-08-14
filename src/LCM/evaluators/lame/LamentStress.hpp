//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LAMENTSTRESS_HPP
#define LAMENTSTRESS_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "lame/LameUtils.hpp"

namespace LCM {
/** \brief Evaluates stress using the Library for Advanced Materials for
 * Engineering with Never-ending Templates (LAMENT).
 */

template <typename EvalT, typename Traits>
class LamentStress : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  LamentStress(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  virtual void
  evaluateFields(typename Traits::EvalData d);

 protected:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGradField;

  std::string                   defGradName, stressName;
  unsigned int                  numQPs;
  unsigned int                  numDims;
  Teuchos::RCP<PHX::DataLayout> tensor_dl;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stressField;

  // The LAMENT material model
  Teuchos::RCP<lament::Material<ScalarT>> lamentMaterialModel;

  // The LAMENT material model name
  std::string lamentMaterialModelName;

  // Vector of the state variable names for the LAMENT material model
  std::vector<std::string> lamentMaterialModelStateVariableNames;

  // Vector of the fields corresponding to the LAMENT material model state
  // variables
  std::vector<PHX::MDField<ScalarT, Cell, QuadPoint>>
      lamentMaterialModelStateVariableFields;
};

}  // namespace LCM

#endif
