//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_WATER_THICKNESS_HPP
#define LANDICE_HYDROLOGY_WATER_THICKNESS_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology Residual Evaluator

    This evaluator computes the hydrology water thickness in the steady case,
    by solving the cavities equation for h
*/

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
class HydrologyWaterThickness : public PHX::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<IsStokes,ScalarT,ParamScalarT>::type      IceScalarT;
  typedef typename std::conditional<ThermoCoupled,ScalarT,ParamScalarT>::type TempScalarT;

  HydrologyWaterThickness (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const IceScalarT>    u_b;
  PHX::MDField<const TempScalarT>   A;
  PHX::MDField<const ScalarT>       m;
  PHX::MDField<const ScalarT>       N;

  // Output:
  PHX::MDField<ScalarT>   h;

  double h_r;
  double l_r;
  double rho_i;
  double c_creep;

  unsigned int numPts;
  std::string   sideSetName;

  bool use_melting;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_WATER_THICKNESS_HPP
