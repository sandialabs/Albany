//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_BASAL_FRICTION_COEFFICIENT_HPP
#define FELIX_BASAL_FRICTION_COEFFICIENT_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Basal friction coefficient evaluator

    This evaluator computes the friction coefficient beta for basal natural BC

*/

template<typename EvalT, typename Traits>
class BasalFrictionCoefficient : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~BasalFrictionCoefficient () {}

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Coefficients for computing beta (if not given)
  double mu;              // Coulomb friction coefficient
  double lambda;          // Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  double power;           // Exponent (for POWER_LAW and REGULARIZED COULOMB only)
  double beta_given_val;  // Constant value (for CONSTANT only)
  double A;               // Constant value for the flowFactorA field (for REGULARIZED_COULOMB only

  // Input:
  PHX::MDField<ParamScalarT>          beta_given_field;
  PHX::MDField<ScalarT>               u_norm;
  PHX::MDField<ParamScalarT>          N;
  PHX::MDField<MeshScalarT>           coordVec;
  PHX::MDField<RealType>              BF;

  // Output:
  PHX::MDField<ScalarT>               beta;

  std::string                     basalSideName;

  bool is_hydrology;
  bool use_stereographic_map;
  double x_0;
  double y_0;
  double R2;

  int numNodes;
  int numQPs;

  enum BETA_TYPE {GIVEN_CONSTANT, GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB, EXP_GIVEN_FIELD};
  BETA_TYPE beta_type;
};

} // Namespace FELIX

#endif // FELIX_BASAL_FRICTION_COEFFICIENT_HPP
