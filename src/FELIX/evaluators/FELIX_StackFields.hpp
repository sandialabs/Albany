//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STACK_RESIDUALS_HPP
#define FELIX_STACK_RESIDUALS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace FELIX {

template<typename EvalT, typename Traits, typename ScalarT>
class StackFieldsBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  // Simple case: 2 vectors
  StackFieldsBase(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl_in1,
                   const Teuchos::RCP<Albany::Layouts>& dl_in2,
                   const Teuchos::RCP<Albany::Layouts>& dl_out);

  // General case
  StackFieldsBase(const Teuchos::ParameterList& p,
                   const std::vector<Teuchos::RCP<Albany::Layouts>>& dls_in,
                   const Teuchos::RCP<Albany::Layouts>& dl_out);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void setup (const Teuchos::ParameterList& p,
              const std::vector<Teuchos::RCP<Albany::Layouts>>& dls_in,
              const Teuchos::RCP<Albany::Layouts>& dl_out);

  Teuchos::RCP<PHX::DataLayout> getLayout (const std::string& name,
                                           const Teuchos::RCP<Albany::Layouts>& dl) const;

  bool isCompatible (const Teuchos::RCP<PHX::DataLayout>& layout_1,
                     const Teuchos::RCP<PHX::DataLayout>& layout_2) const;


  // Input:
  std::vector<PHX::MDField<const ScalarT>> fields_in;

  // Output:
  PHX::MDField<ScalarT> field_out;

  int              num_fields_in;
  int              rank_out;
  std::vector<int> dims_out;
  std::vector<int> dims_in;
  std::vector<int> offsets;
  std::vector<int> ranks_in;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using StackFields = StackFieldsBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using StackFieldsMesh = StackFieldsBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using StackFieldsParam = StackFieldsBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace FELIX

#endif // FELIX_STACK_RESIDUALS_HPP
