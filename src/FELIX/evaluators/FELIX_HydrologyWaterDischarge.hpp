//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
#define FELIX_HYDROLOGY_WATER_DISCHARGE_HPP 1

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
class HydrologyWaterDischarge : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ScalarT>  gradN;
  PHX::MDField<ScalarT>  gradPhiH;
  PHX::MDField<ScalarT>  h;

  // Output:
  PHX::MDField<ScalarT>  q;

  int numQPs;
  int numDim;

  bool          stokes_coupling;
  std::string   sideSetName;

  double mu_w;
  double k;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
