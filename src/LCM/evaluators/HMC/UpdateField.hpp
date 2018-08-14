//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LCM_UPDATEFIELD_HPP
#define LCM_UPDATEFIELD_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

namespace LCM {
/** \brief UpdateField

*/

template <typename EvalT, typename Traits>
class UpdateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  UpdateField(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<ScalarT>       field_Nplus1;
  PHX::MDField<const ScalarT> field_Inc;

  std::string name_N;
};
}  // namespace LCM

#endif
