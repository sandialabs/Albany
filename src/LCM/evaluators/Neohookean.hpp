//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Neohookean_hpp)
#define LCM_Neohookean_hpp

#include <Intrepid2_MiniTensor.h>
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief Neohookean stress response
  ///
  /// This evaluator computes stress based on a uncoupled Neohookean
  /// Helmholtz potential
  /// \f$ \sigma_{ij} = \frac{\kappa}{2}(J-1/J)\delta_{ij}
  /// + \mu dev(\bar{b_{ij}}) \f$
  template<typename EvalT, typename Traits>
  class Neohookean : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    Neohookean(const Teuchos::ParameterList& p,
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
    /// Input: Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad;

    ///
    /// Input: Determinant of Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> J;

    ///
    /// Input: Elastic (or Young's) Modulus
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;

    ///
    /// Input: Poisson's Ratio
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

    ///
    /// Output: Cauchy Stress
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

    ///
    /// Number of integration points
    ///
    unsigned int numQPs;

    ///
    /// Number of problem dimensions
    ///
    unsigned int numDims;

    ///
    /// Number of elements in workset
    ///
    unsigned int worksetSize;

    ///
    /// Local tensors for computation
    ///
    Intrepid2::Tensor<ScalarT> F,b,sigma,I;
  };
}

#endif
