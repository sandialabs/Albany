//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_BASAL_FRICTION_COEFFICIENT_GRADIENT_HPP
#define LANDICE_BASAL_FRICTION_COEFFICIENT_GRADIENT_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Basal friction coefficient evaluator

    This evaluator computes the friction coefficient beta for basal natural BC

*/

template<typename EvalT, typename Traits>
class BasalFrictionCoefficientGradient : public PHX::EvaluatorWithBaseImpl<Traits>,
                                         public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;

  BasalFrictionCoefficientGradient (const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~BasalFrictionCoefficientGradient () {}

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

  typedef typename PHX::Device execution_space;

private:

  void computeEffectivePressure (int cell, int side);

  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  // Input:
  // TODO: restore layout template arguments when removing old sideset layout
  PHX::MDField<const ParamScalarT>      given_field;       // Side, Node
  PHX::MDField<const ParamScalarT>  given_field_param; // Side, Node
  PHX::MDField<const MeshScalarT>   GradBF;            // Side, Node, QuadPoint, Dim
  PHX::MDField<const ParamScalarT>  N;                 // Side, QuadPoint
  PHX::MDField<const ScalarT>       U;                 // Side, QuadPoint, Dim
  PHX::MDField<const ScalarT>       gradN;             // Side, QuadPoint, Dim
  PHX::MDField<const ScalarT>       gradU;             // Side, QuadPoint, Dim, Dim
  PHX::MDField<const ScalarT>       u_norm;            // Side, QuadPoint
  PHX::MDField<const MeshScalarT>   coordVec;          // Side, QuadPoint, Dim

  PHX::MDField<const ScalarT,Dim>                               lambdaParam;
  PHX::MDField<const ScalarT,Dim>                               muParam;
  PHX::MDField<const ScalarT,Dim>                               powerParam;

  // Output:
  PHX::MDField<ScalarT>           grad_beta;           // Side, QuadPoint, Dim

  std::string basalSideName;
  Albany::LocalSideSetInfo sideSet;

  unsigned int numSideNodes;
  unsigned int numSideQPs;
  unsigned int sideDim;
  unsigned int vecDim;

  double A;
  double x_0;
  double y_0;
  double R2;

  ScalarT lambda;
  ScalarT mu;
  ScalarT power;

  bool use_stereographic_map;
  bool is_given_field_param;

  enum BETA_TYPE {INVALID, GIVEN_CONSTANT, GIVEN_FIELD, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct GivenFieldParam_Tag{};
  struct GivenField_Tag{};
  struct RegularizedCoulomb_Tag{};
  struct StereographicMapCorrection_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,GivenFieldParam_Tag> GivenFieldParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,GivenField_Tag> GivenField_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,RegularizedCoulomb_Tag> RegularizedCoulomb_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,StereographicMapCorrection_Tag> StereographicMapCorrection_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const GivenFieldParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const GivenField_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const RegularizedCoulomb_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const StereographicMapCorrection_Tag& tag, const int& i) const;

};

} // Namespace LandIce

#endif // LANDICE_BASAL_FRICTION_COEFFICIENT_GRADIENT_HPP
