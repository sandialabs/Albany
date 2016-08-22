//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_MELTING_RATE_HPP
#define FELIX_HYDROLOGY_MELTING_RATE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, bool IsStokes>
class HydrologyMeltingRate : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  typedef typename std::conditional<IsStokes,ScalarT,ParamScalarT>::type     IceScalarT;

  HydrologyMeltingRate (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<IceScalarT>    u_b;
  PHX::MDField<ScalarT>       beta;
  PHX::MDField<ParamScalarT>  G;

  // Output:
  PHX::MDField<ScalarT>       m;

  int               numQPs;
  double            L;

  std::string       sideSetName; // Only needed if IsStokes=true
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_MELTING_RATE_HPP
