//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_FIELD_NORM_HPP
#define FELIX_FIELD_NORM_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits>
class FieldNorm : public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  FieldNorm (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  ScalarT   regularizationParam;
  ScalarT*  regularizationParamPtr;
  ScalarT   printedH;

  // Input:
  PHX::MDField<ScalarT> field;

  // Output:
  PHX::MDField<ScalarT> field_norm;

  std::string sideSetName;
  std::vector<PHX::DataLayout::size_type> dims;
  int numDims;
};

} // Namespace FELIX

#endif // FELIX_FIELD_NORM_HPP
