//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_ICE_OVERBURDEN_HPP
#define LANDICE_ICE_OVERBURDEN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Ice overburden

    This evaluator evaluates the ice overburden P_o = rho_i*g*H,
    with H being the ice thickness
*/

template<typename EvalT, typename Traits, bool IsStokes>
class IceOverburden : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  IceOverburden (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const ParamScalarT>  H;

  // Output:
  PHX::MDField<ParamScalarT>        P_o;

  std::string basalSideName;  // Only if IsStokes  is true

  int numPts;

  double rho_i;
  double g;
};

} // Namespace LandIce

#endif // LANDICE_ICE_OVERBURDEN_HPP
