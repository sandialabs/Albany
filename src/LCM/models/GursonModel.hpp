//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(GursonModel_hpp)
#define GursonModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

//! \brief Gurson Finite Deformation Model
template <typename EvalT, typename Traits>
class GursonModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT                             ScalarT;
  typedef typename EvalT::MeshScalarT                         MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  ///
  /// Constructor
  ///
  GursonModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~GursonModel(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

 private:
  ///
  /// Private to prohibit copying
  ///
  GursonModel(const GursonModel&);

  ///
  /// Private to prohibit copying
  ///
  GursonModel&
  operator=(const GursonModel&);

  ///
  /// Saturation hardening constants
  ///
  RealType sat_mod_, sat_exp_;

  ///
  /// Initial Void Volume
  ///
  RealType f0_;

  ///
  /// Shear Damage Parameter
  ///
  RealType kw_;

  ///
  /// Void Nucleation Parameters
  ///
  RealType eN_, sN_, fN_;

  ///
  /// Critical Void Parameters
  ///
  RealType fc_, ff_;

  ///
  /// Yield Parameters
  ///
  RealType q1_, q2_, q3_;

  ///
  /// Compute Yield Function
  ///
  ScalarT
  YieldFunction(
      minitensor::Tensor<ScalarT> const& s,
      ScalarT const&                     p,
      ScalarT const&                     fvoid,
      ScalarT const&                     eq,
      ScalarT const&                     K,
      ScalarT const&                     Y,
      ScalarT const&                     jacobian,
      ScalarT const&                     E);

  ///
  /// Compute Residual and Local Jacobian
  ///
  void
  ResidualJacobian(
      std::vector<ScalarT>&        X,
      std::vector<ScalarT>&        R,
      std::vector<ScalarT>&        dRdX,
      const ScalarT&               p,
      const ScalarT&               fvoid,
      const ScalarT&               eq,
      minitensor::Tensor<ScalarT>& s,
      const ScalarT&               mu,
      const ScalarT&               kappa,
      const ScalarT&               K,
      const ScalarT&               Y,
      const ScalarT&               jacobian);
};
}  // namespace LCM

#endif
