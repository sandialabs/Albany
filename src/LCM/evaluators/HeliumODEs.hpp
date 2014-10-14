//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(HeliumODEs_hpp)
#define HeliumODEs_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  /// This evaluator integrates coupled ODEs for the
  ///   1. He concentration
  ///   2. Total bubble density
  ///   3. Bubble volume fraction
  /// We employ implicit integration (backward Euler)
  ///
  template<typename EvalT, typename Traits>
  class HeliumODEs : public PHX::EvaluatorWithBaseImpl<Traits>,
                                       public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    HeliumODEs(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Input: total_concentration - addition of lattice and trapped
    ///        concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> total_concentration_;

    ///
    /// Input: time step
    ///
    PHX::MDField<ScalarT,Dummy> delta_time_;

    ///
    /// Input: temperature dependent diffusion coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> diffusion_coefficient_;

    ///
    /// Input: spatial dimension and number of integration points
    ///
    std::size_t num_dims_;
    std::size_t num_pts_;

    ///
    /// Output
    /// (1) Helium concentration
    /// (2) Total bubble density
    /// (3) Bubble volume fracture
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> he_concentration_;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> total_bubble_density_;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> bubble_volume_fraction_;

    ///
    /// Constants
    ///
    RealType avogadros_num_, omega_, t_decay_constant_, he_radius_, eta_;

    /// 
    /// Scalar names for obtaining state old
    ///

    std::string total_concentration_name_;
    std::string he_concentration_name_;
    std::string total_bubble_density_name_;
    std::string bubble_volume_fraction_name_;

  };
}

#endif
