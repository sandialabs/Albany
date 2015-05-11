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
  typedef typename Sacado::Fad::DFad<ScalarT> Fad;

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
  /// explicit update of the slip
  ///
  void
  updateSlipViaExplicitIntegration(ScalarT                        dt,
				   std::vector<ScalarT> const &   slip_n,
				   std::vector<ScalarT> const &   hardness_n,
				   Intrepid::Tensor<Fad> const &  S,
				   std::vector<Fad> &             slip_np1);

  ///
  /// Compute Lp_np1 and Fp_np1 based on computed slip increment
  ///
  void
  applySlipIncrement(std::vector<ScalarT> const &       slip_n,
		     std::vector<Fad> const &           slip_np1,
		     Intrepid::Tensor<ScalarT> const &  Fp_n,
		     Intrepid::Tensor<Fad> &            Lp_np1,
		     Intrepid::Tensor<Fad> &            Fp_np1);

  ///
  /// update the hardness
  ///
  void
  updateHardness(std::vector<Fad> const &      slip_np1,
		 std::vector<ScalarT> const &  hardness_n,
		 std::vector<Fad> &            hardness_np1);
  
  ///
  /// residual
  ///
  void
  computeResidual(ScalarT                       dt,
		  std::vector<ScalarT> const &  slip_n,
		  std::vector<Fad> const &      slip_np1,
		  std::vector<Fad> const &      hardness_np1,
		  std::vector<Fad> const &      shear_np1,
		  std::vector<Fad> &            slip_residual,
		  Fad &                         norm_slip_residual);

  ///
  /// compute stresses
  ///
  void 
  computeStress(Intrepid::Tensor<ScalarT> const &  F,
                Intrepid::Tensor<Fad> const &      Fp,
                Intrepid::Tensor<Fad> &            T,
                Intrepid::Tensor<Fad> &            S,
		std::vector<Fad>      &            shear);

  void
  constructMatrixFiniteDifference(ScalarT                            dt,
				  Intrepid::Tensor<ScalarT> const &  Fp_n,
				  Intrepid::Tensor<ScalarT> const &  F_np1,
				  std::vector<ScalarT> const &       slip_n,
				  std::vector<Fad> const &           slip_np1,
				  std::vector<ScalarT> const &       hardness_n,
				  std::vector<Fad> &                 matrix);

  ///
  /// Check tensor for nans and infs.
  ///
  void confirmTensorSanity(Intrepid::Tensor<Fad> const & input,
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

    SlipSystemStruct() {}

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
  };
}

#endif
