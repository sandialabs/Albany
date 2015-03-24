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

  ~BasalFrictionCoefficient ();

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void setHomotopyParamPtr(ScalarT* h);

  void evaluateFields (typename Traits::EvalData d);

  typedef typename PHX::Device execution_space;

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  ScalarT* homotopyParam;
  ScalarT printedH;

  double getScalarTValue (const ScalarT& s);

  // Coefficients for computing beta (if not given)
  double mu;    // Coulomb friction coefficient
  double N;     // Effective pressure
  int    n;     // Glen's law exponent
  double L;     // Roughness of the bed (for REGULARIZED_COULOMB only)
  double power; // Exponent (for POWER_LAW only)

  // Data to compute beta in case beta(|u|) is a piecewise linear function of |u|
  int      nb_pts;
  double*  u_grid;
  double*  u_grid_h;
  double*  beta_coeffs;

  // Input:
  PHX::MDField<ScalarT,Cell,Node,Dim> velocity;
  PHX::MDField<ScalarT,Cell,Node>     beta_given;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> beta;

  unsigned int numDims, numNodes, numCells;

  enum BETA_TYPE {FROM_FILE, UNIFORM, POWER_LAW, REGULARIZED_COULOMB, PIECEWISE_LINEAR};
  BETA_TYPE beta_type;
};

template<>
double BasalFrictionCoefficient<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>::getScalarTValue(const ScalarT& s);

} // Namespace FELIX

#endif // FELIX_BASAL_FRICTION_COEFFICIENT_HPP
