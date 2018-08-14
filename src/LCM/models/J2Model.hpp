//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2Model_hpp)
#define LCM_J2Model_hpp

#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

//! \brief J2 Plasticity Constitutive Model
template <typename EvalT, typename Traits>
class J2Model : public LCM::ConstitutiveModel<EvalT, Traits>
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

  // optional temperature support
  using ConstitutiveModel<EvalT, Traits>::have_temperature_;
  using ConstitutiveModel<EvalT, Traits>::expansion_coeff_;
  using ConstitutiveModel<EvalT, Traits>::ref_temperature_;
  using ConstitutiveModel<EvalT, Traits>::heat_capacity_;
  using ConstitutiveModel<EvalT, Traits>::density_;
  using ConstitutiveModel<EvalT, Traits>::temperature_;

  ///
  /// Constructor
  ///
  J2Model(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual ~J2Model(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

 private:
  ///
  /// Private to prohibit copying
  ///
  J2Model(const J2Model&);

  ///
  /// Private to prohibit copying
  ///
  J2Model&
  operator=(const J2Model&);

  ///
  /// Saturation hardening constants
  ///
  RealType sat_mod_, sat_exp_;

  // Kokkos
  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);
};
}  // namespace LCM

#endif
