//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_TvergaardHutchinsonModel_hpp)
#define LCM_TvergaardHutchinsonModel_hpp

#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

//! \brief TvergaardHutchinson Model
template <typename EvalT, typename Traits>
class TvergaardHutchinsonModel : public LCM::ConstitutiveModel<EvalT, Traits>
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
  using ConstitutiveModel<EvalT, Traits>::compute_energy_;
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

  ///
  /// Constructor
  ///
  TvergaardHutchinsonModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~TvergaardHutchinsonModel(){};

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
  TvergaardHutchinsonModel(const TvergaardHutchinsonModel&);

  ///
  /// Private to prohibit copying
  ///
  TvergaardHutchinsonModel&
  operator=(const TvergaardHutchinsonModel&);

  ///
  /// Constants
  ///
  RealType delta_1, delta_2, delta_c, sigma_c, beta_0, beta_1, beta_2;
};
}  // namespace LCM

#endif
