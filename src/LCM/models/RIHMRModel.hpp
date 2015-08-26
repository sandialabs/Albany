//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(RIHMRModel_hpp)
#define RIHMRModel_hpp

#include <Intrepid_MiniTensor.h>
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM
{

//! \brief Rate Independent Hardening Minus Recovery (RIHMR) Model
template<typename EvalT, typename Traits>
class RIHMRModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  ///
  /// Constructor
  ///
  RIHMRModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~RIHMRModel()
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
  RIHMRModel(const RIHMRModel&);

  ///
  /// Private to prohibit copying
  ///
  RIHMRModel& operator=(const RIHMRModel&);

  ///
  /// Saturation hardening constants
  ///
  RealType sat_mod_, sat_exp_;

  ///
  /// Compute Residual and Local Jacobian
  ///
  void
  ResidualJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
      std::vector<ScalarT> & dRdX, const ScalarT & es, const ScalarT & smag,
      const ScalarT & mubar, ScalarT & mu, ScalarT & kappa, ScalarT & K,
      ScalarT & Y, ScalarT & Rd);
};
}

#endif
