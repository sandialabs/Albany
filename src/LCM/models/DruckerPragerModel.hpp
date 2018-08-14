//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_DruckerPragerModel_hpp)
#define LCM_DruckerPragerModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado.hpp"

namespace LCM {

///
/// \brief Constitutive Model Base Class
///
///

template <typename EvalT, typename Traits>
class DruckerPragerModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT                             ScalarT;
  typedef typename EvalT::MeshScalarT                         MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;

  using Base::field_name_map_;
  using Base::num_dims_;
  using Base::num_pts_;

  ///
  /// Constructor
  ///
  DruckerPragerModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~DruckerPragerModel(){};

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
  DruckerPragerModel(const DruckerPragerModel&);

  ///
  /// Private to prohibit copying
  ///
  DruckerPragerModel&
  operator=(const DruckerPragerModel&);

  ///
  /// Parameters for hardening law
  ///
  RealType a0_, a1_, a2_, a3_, b0_;

  ///
  /// Cohesion-like parameters
  ///
  RealType Cf_, Cg_;

  ///
  /// Compute residual and local jacobian
  ///
  void
  ResidualJacobian(
      std::vector<ScalarT>& X,
      std::vector<ScalarT>& R,
      std::vector<ScalarT>& dRdX,
      const ScalarT         ptr,
      const ScalarT         qtr,
      const ScalarT         eqN,
      const ScalarT         mu,
      const ScalarT         kappa);
};
}  // namespace LCM

#endif
