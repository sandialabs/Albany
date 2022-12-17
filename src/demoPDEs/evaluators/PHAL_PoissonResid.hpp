//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_POISSON_RESID_HPP
#define PHAL_POISSON_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
class PoissonResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  PoissonResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT> wBF;
  PHX::MDField<const ScalarT> Potential;
  PHX::MDField<const ScalarT> Permittivity;
  PHX::MDField<const MeshScalarT> wGradBF;
  PHX::MDField<const ScalarT> PhiGrad;
  PHX::MDField<const ScalarT> Source;

  bool haveSource;

  // Output:
  PHX::MDField<ScalarT> PhiResidual;
  PHX::MDField<ScalarT> PhiFlux;
};

} // namespace PHAL

#endif // PHAL_POISSON_RESID_HPP
