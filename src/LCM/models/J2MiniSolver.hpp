//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2MiniSolver_hpp)
#define LCM_J2MiniSolver_hpp

#include "ConstitutiveModel.hpp"

namespace LCM
{

//! \brief J2 Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class J2MiniSolver: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

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
  J2MiniSolver(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// No copy constructor
  ///
  J2MiniSolver(J2MiniSolver const &) = delete;

  ///
  /// No copy assignment
  ///
  J2MiniSolver & operator=(J2MiniSolver const &) = delete;

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~J2MiniSolver()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

private:

  ///
  /// Saturation hardening constants
  ///
  RealType
  sat_mod_;

  RealType
  sat_exp_;

 //Kokkos 
  virtual
  void
  computeStateParallel(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);
};

}
#endif // LCM_J2MiniSolver_hpp
