//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_HyperelasticDamageModel_hpp)
#define LCM_HyperelasticDamageModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"

namespace LCM
{

//! \brief Hyperelasticity with isotropic damage
template<typename EvalT, typename Traits>
class HyperelasticDamageModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

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

  // optional damage support
  using ConstitutiveModel<EvalT, Traits>::damage_;
  using ConstitutiveModel<EvalT, Traits>::have_damage_;

  ///
  /// Constructor
  ///
  HyperelasticDamageModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~HyperelasticDamageModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields){
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
 }


private:

  ///
  /// Private to prohibit copying
  ///
  HyperelasticDamageModel(const HyperelasticDamageModel&);

  ///
  /// Private to prohibit copying
  ///
  HyperelasticDamageModel& operator=(const HyperelasticDamageModel&);

  ///
  /// Max Damage
  ///
  RealType max_damage_;

  ///
  /// Damage Saturation
  ///
  RealType damage_saturation_;

};
}

#endif
