//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid_MiniTensor.h>

namespace LCM
{

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  ///
  /// Constructor
  ///
  CrystalPlasticityModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~CrystalPlasticityModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields){
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
 }


private:

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel(const CrystalPlasticityModel&);

  ///
  /// Private to prohibit copying
  ///
  CrystalPlasticityModel& operator=(const CrystalPlasticityModel&);

  ///
  /// predictor
  ///
  void
  predictor(int cell,
	    int pt,
	    ScalarT dt,
	    std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > & slips,
	    std::vector<Albany::MDArray * > const & previous_slips,
	    std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > & hards,
	    std::vector<Albany::MDArray * > const & previous_hards,
	    Intrepid::Tensor<ScalarT> const & F,
	    Intrepid::Tensor<ScalarT> & L,
	    Intrepid::Tensor<ScalarT> & Fp);

  ///
  /// helper
  ///
  void 
  computeStress(Intrepid::Tensor<ScalarT> const & F,
                Intrepid::Tensor<ScalarT> const & Fp,
                Intrepid::Tensor<ScalarT>       & T,
                Intrepid::Tensor<ScalarT>       & S);

  ///
  /// Check tensor for nans and infs.
  ///
  void confirmTensorSanity(Intrepid::Tensor<ScalarT> const & input,
			   std::string const & message);

  ///
  /// Crystal elasticity parameters
  ///
  RealType c11_,c12_,c44_;
  Intrepid::Tensor4<RealType> C_;
  Intrepid::Tensor<RealType> orientation_;
 
  ///
  /// Number of slip systems
  ///
  int num_slip_;

  //! \brief Struct to slip system information
  struct SlipSystemStruct {

    SlipSystemStruct() {
      // s_ = Intrepid::Vector<RealType> (num_dims_, Intrepid::ZEROS);
      // n_ = Intrepid::Vector<RealType> (num_dims_, Intrepid::ZEROS);
      // tau_critical_ = 1.0;
      // gamma_dot_0_  = 0.0;
      // gamma_exp_    = 0.0;
      // H_            = 0.0;
    }

    // slip system vectors
    Intrepid::Vector<RealType> s_, n_;

    // Schmid Tensor
    Intrepid::Tensor<RealType> projector_;

    // flow rule parameters
    RealType tau_critical_, gamma_dot_0_, gamma_exp_, H_;
  };

  ///
  /// Crystal Plasticity parameters
  ///
  std::vector<SlipSystemStruct> slip_systems_;


  ///
  /// Workspace
  ///
  Intrepid::Tensor<ScalarT> F_, Fpinv_, Fe_, E_; 
  Intrepid::Tensor<RealType> I_;
  };
}

#endif
