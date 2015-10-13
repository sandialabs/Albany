//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
  typedef typename EvalT::ScalarT ScalarT;

  BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~BasalFrictionCoefficient () {}

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void setHomotopyParamPtr(ScalarT* h);

  void evaluateFields (typename Traits::EvalData d);

  typedef typename PHX::Device execution_space;

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  ScalarT* homotopyParam;

  // Coefficients for computing beta (if not given)
  double mu;              // Coulomb friction coefficient
  double lambda;          // Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  double power;           // Exponent (for POWER_LAW and REGULARIZED COULOMB only)
  double beta_given_val;  // Constant value (for CONSTANT only)
  double A;               // Constant value for the flowFactorA field (for REGULARIZED_COULOMB only

  // Input:
//  PHX::MDField<ScalarT,Cell>                          flowFactorA;       //this is the coefficient A of the flow factor
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>           u_norm;
  PHX::MDField<ScalarT,Cell,Node>                     beta_given_field;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>           N;

  PHX::MDField<MeshScalarT,Cell,Side,Node,QuadPoint>  BF;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>     beta;

  std::string                     basalSideName;
  std::vector<std::vector<int> >  sideNodes;     // Needed only in case of given beta

  int numSideNodes;
  int numSideQPs;

  enum BETA_TYPE {GIVEN_CONSTANT, GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;
};

} // Namespace FELIX

#endif // FELIX_BASAL_FRICTION_COEFFICIENT_HPP
