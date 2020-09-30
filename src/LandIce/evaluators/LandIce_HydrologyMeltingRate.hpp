//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_MELTING_RATE_HPP
#define LANDICE_HYDROLOGY_MELTING_RATE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology Melting Rate evaluator

    This evaluator evaluates the following:

      m = (G + \beta |ub|^2) /L

    where
      - G: geothermal flux
      - \beta: basal friction parameter in ice b.c. : sigma*n + \beta u = 0
      - ub: sliding velocity
      - L: ice latent heat
    See below for units
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
  PHX::MDField<const IceScalarT>    u_b;    // [ m yr^-1 ]
  PHX::MDField<const ScalarT>       beta;   // [ kPa yr m^-1 ]
  PHX::MDField<const ParamScalarT>  G;      // [ W m^-2 ]

  // Output:
  PHX::MDField<ScalarT>             m;      // [ kg m^-3 m yr^-1 ], that is, [ kg m^-2 yr ^-1 ] (density*speed)

  bool              nodal;

  unsigned int               numQPs;
  unsigned int               numNodes;
  double            L;            // Ice Latent Heat [J kg^-1]
  double            scaling_G;    // Used internally

  std::string       sideSetName;  // Only needed if IsStokes=true
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_MELTING_RATE_HPP
