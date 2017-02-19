//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_PROLONGATE_VECTOR_HPP
#define FELIX_PROLONGATE_VECTOR_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace FELIX {
/** \brief Prolongation Evaluator

    This evaluator prolongates a vector/gradient layout to one of bigger size

*/

template<typename EvalT, typename Traits, typename ScalarT>
class ProlongateVectorBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  ProlongateVectorBase(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl_in,
                       const Teuchos::RCP<Albany::Layouts>& dl_out);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void setup (const Teuchos::ParameterList& p,
              const std::vector<Teuchos::RCP<Albany::Layouts>>& dls_in,
              const Teuchos::RCP<Albany::Layouts>& dl_out);

  // Input:
  PHX::MDField<const ScalarT> v_in;

  // Output:
  PHX::MDField<ScalarT> v_out;

  bool pad_back;
  ScalarT pad_value;

  std::vector<int> dims_in;
  std::vector<int> dims_out;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using ProlongateVector = ProlongateVectorBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ProlongateVectorMesh = ProlongateVectorBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ProlongateVectorParam = ProlongateVectorBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace FELIX

#endif // FELIX_PROLONGATE_VECTOR_HPP
