//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Author: Mario J. Juha (juham@rpi.edu)

#if !defined(LCM_ViscoElasticModel_hpp)
#define LCM_ViscoElasticModel_hpp

#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"
#include "MiniTensor.h"
#include "MiniTensor_Definitions.h"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

namespace VE {
// Do not change it!
static constexpr minitensor::Index MAX_DIM = 3;
}  // namespace VE

//! \brief ElasticCrystal Model
template <typename EvalT, typename Traits>
class ViscoElasticModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // accessing directly protected variables of Constitutive model.
  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::compute_energy_;
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

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
  ViscoElasticModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~ViscoElasticModel(){
      // empty
  };

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  // This capability not implemented yet.
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
  ViscoElasticModel(const ViscoElasticModel&);

  ///
  /// Private to prohibit copying
  ///
  ViscoElasticModel&
  operator=(const ViscoElasticModel&);

  // Relaxation time
  RealType tau1_, tau2_, tau3_;

  // Stiffness ratio.
  RealType gamma1_, gamma2_, gamma3_, gamma_inf_;

  // Shear modulus
  RealType mu_;

  // Gas constant
  RealType R_;
};
}  // namespace LCM

#endif
