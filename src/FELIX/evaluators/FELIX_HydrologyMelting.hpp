//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_MELTING_HPP
#define FELIX_HYDROLOGY_MELTING_HPP 1

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

template<typename EvalT, typename Traits>
class HydrologyMelting : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologyMelting (const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ParamScalarT>      u_b;
  PHX::MDField<ParamScalarT>      beta;
  PHX::MDField<ParamScalarT>      G;

  // Output:
  PHX::MDField<ParamScalarT>      m;

  bool              stokes_coupling;
  std::string       sideSetName;

  int numQPs;
  int numDim;

  double L;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_MELTING_HPP
