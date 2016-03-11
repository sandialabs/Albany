//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>  gradPhi;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>  q;
  PHX::MDField<ParamScalarT,Cell,QuadPoint>  u_b;
  PHX::MDField<ParamScalarT,Cell,QuadPoint>  beta;
  PHX::MDField<ParamScalarT,Cell,QuadPoint>  G;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint>      m;

  int numQPs;
  int numDim;

  double nonlin_coeff;
  double mu_w;
  double L;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_MELTING_HPP
