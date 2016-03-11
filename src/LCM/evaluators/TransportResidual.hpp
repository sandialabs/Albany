//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_TransportResidual_hpp)
#define LCM_TransportResidual_hpp

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  ///  This evaluator computes the residual for the transport equation
  ///
  template<typename EvalT, typename Traits>
  class TransportResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    TransportResidual(Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

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
    /// Scalar field for transport variable
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> scalar_;

    ///
    /// Scalar dot field for transport variable
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> scalar_dot_;

    ///
    /// Scalar field for transport variable
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint, Dim> scalar_grad_;

    ///
    /// Integrations weights
    ///
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights_;

    ///
    /// Weighted basis functions
    ///
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> w_bf_;

    ///
    /// Weighted gradients of basis functions
    ///
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> w_grad_bf_;

    ///
    /// Source term(s)
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> source_;
    PHX::MDField<ScalarT,Cell,QuadPoint> second_source_;

    ///
    /// M operator for contact
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> M_operator_;

    ///
    /// Scalar coefficient on the transient transport term
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> transient_coeff_;

    ///
    /// Tensor diffusivity
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> diffusivity_;

    ///
    /// Vector convection term
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> convection_vector_;

    ///
    /// Species coupling term
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> species_coupling_;

    ///
    /// Stabilization term
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> stabilization_;

    ///
    /// Output residual
    ///
    PHX::MDField<ScalarT,Cell,Node> residual_;

    ///
    ///  Feature flags
    ///
    bool have_source_;
    bool have_second_source_;
    bool have_transient_;
    bool have_diffusion_;
    bool have_convection_;
    bool have_species_coupling_;
    bool have_stabilization_;
    bool have_contact_;

    ///
    /// Data structure dimensions
    ///
    int num_nodes_, num_pts_, num_dims_;

    ///
    /// Scalar name
    ///
    std::string scalar_name_;

  };
}

#endif
