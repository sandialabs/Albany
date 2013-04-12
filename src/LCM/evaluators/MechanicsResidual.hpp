//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Mechanics_Residual_hpp)
#define LCM_Mechanics_Residual_hpp

#include <Intrepid_MiniTensor_Tensor.h>

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  ///
  /// \brief Mechanics Residual
  ///
  /// This evaluator computes the residual due to the balance
  /// of linear momentum for infinitesimal and finite deformation,
  /// with or without dynamics
  ///
  template<typename EvalT, typename Traits>
  class MechanicsResidual : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

  public:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Constructor
    ///
    MechanicsResidual(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void 
    postRegistrationSetup(typename Traits::SetupData d,
                          PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void 
    evaluateFields(typename Traits::EvalData d);

    ///
    /// Sacado::Parameter method
    ///
    ScalarT& 
    getValue(const std::string &n);

  private:

    ///
    /// Input: Cauchy Stress
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

    ///
    /// Input: Determinant of the Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> J;

    ///
    /// Input: Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;

    ///
    /// Input: Weighted Basis Function Gradients
    ///
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

    ///
    /// Input: Weighted Basis Functions
    ///
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

    ///
    /// Input: gravity constant
    ///
    ScalarT zGrav;

    ///
    /// Optional
    /// Input: Pore Pressure
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> porePressure;

    ///
    /// Optional
    /// Input: Biot Coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> biotCoeff;

    ///
    /// Output: Residual Forces
    ///
    PHX::MDField<ScalarT,Cell,Node,Dim> Residual;

    ///
    /// Number of element nodes
    ///
    std::size_t numNodes;

    ///
    /// Number of integration points
    ///
    std::size_t numQPs;

    ///
    /// Number of spatial dimensions
    ///
    std::size_t numDims;

    ///
    /// Pore Pressure flag
    ///
    bool havePorePressure;

    ///
    /// Tensors for local computations
    ///
    Intrepid::Tensor<ScalarT> F, P, sig, I;

  };
}

#endif
