//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ElasticDamageModel_hpp)
#define LCM_ElasticDamageModel_hpp

#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

//! \brief Constitutive Model Base Class
template <typename EvalT, typename Traits>
class ElasticDamageModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  // optional material tangent computation
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

  ///
  /// Constructor
  ///
  ElasticDamageModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructorstructor
  ///
  virtual ~ElasticDamageModel(){};

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
  ElasticDamageModel(const ElasticDamageModel&);

  ///
  /// Private to prohibit copying
  ///
  ElasticDamageModel&
  operator=(const ElasticDamageModel&);

  ///
  /// Damage parameters
  ///
  RealType max_damage_, saturation_;
};
}  // namespace LCM

#endif
