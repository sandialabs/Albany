//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_WATER_DISCHARGE_HPP
#define LANDICE_HYDROLOGY_WATER_DISCHARGE_HPP 1

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"
#include "PHAL_Dimension.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/* \brief Hydrology Water Discharge Evaluator
 *
 *   This evaluator evaluates
 *
 *     q = - (k_0 * h^3)/(\rho_w * g) * grad \Phi
 *
 *   It is assumed that the units of each term are
 *
 *    1) k_0       : [m^-1 s^-1]
 *    2) h         : [m]
 *    3) \rho_w    : [kg m^-3]
 *    4) g         : [m s^-2]
 *    5) grad \Phi : [kPa km^-1]
 *
 *   which yields water discharge units [m^2 s^-1].
 *
 */

template<typename EvalT, typename Traits>
class HydrologyWaterDischarge : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const ScalarT>       gradPhi;
  PHX::MDField<const ScalarT>       gradPhiNorm;
  PHX::MDField<const ScalarT>       h;
  PHX::MDField<const ScalarT,Dim>   regularizationParam;
  PHX::MDField<const ScalarT,Dim>   k_param;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  // Output:
  PHX::MDField<ScalarT>   q;

  unsigned int numQPs;
  unsigned int numDim;

  bool eval_on_side;
  std::string   sideSetName;  // Only used if eval_on_side=true
  Albany::LocalSideSetInfo sideSet; // Needed only if eval_on_side=true

  double alpha;
  double beta;

  enum RegularizationType { NONE=1, GIVEN_VALUE, GIVEN_PARAMETER};

  RegularizationType reg_type;
  ScalarT regularization;
  ScalarT printedReg;
  ScalarT printedKappa;

  bool needsGradPhiNorm;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_WATER_DISCHARGE_HPP
