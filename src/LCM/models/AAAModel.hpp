//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_AAAModel_hpp)
#define LCM_AAAModel_hpp


#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM {
/** \brief Nearly Incompressible AAA model

    This evaluator computes the Cauchy stress based on a decoupled
    Helmholtz potential. Models a hyperelastic material for use in
    modeling Abdominal Aortic Aneurysm (AAA).

    Material model is given in Raghaven and Vorp, Journal of
    Biomechanics 33 (2000) 475-482.

    Special case of the generalized power law neo-Hookean
    model given by eg Zhang and Rajagopal, 1992.
 */

template<typename EvalT, typename Traits>
class AAAModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  /// Constructor
  AAAModel(Teuchos::ParameterList* p,
	const Teuchos::RCP<Albany::Layouts>& dl);

  /// Virtual Destructor
  virtual
  ~AAAModel()
  {};

  /// Method to compute the state
  virtual
  void
  computeState(typename Traits::EvalData workset,
		  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
		  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields);

  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
                  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
                  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields){
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
 }

private:

  /// Private to prohibit copying
  AAAModel(const AAAModel&);

  /// Private to prohibit copying
  AAAModel& operator=(const AAAModel&);

  /// Material parameters
  RealType alpha_;
  RealType beta_;
  RealType mult_;
};

} // namespace LCM

#endif /* LCM_AAAmodel_hpp */
