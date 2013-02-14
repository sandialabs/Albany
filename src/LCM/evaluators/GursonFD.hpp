//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef GURSONFD_HPP
#define GURSONFD_HPP

#include <Intrepid_MiniTensor.h>
#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Sacado.hpp"

namespace LCM {
  ///\brief Gurson large deformation hyperelastic stress response
  ///
  /// This evaluator computes stress based on original Gurson model.
  ///

  template<typename EvalT, typename Traits>
  class GursonFD: public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    ///
    /// Constructor
    ///
    GursonFD(const Teuchos::ParameterList& p,
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
    typedef typename Sacado::Fad::DFad<ScalarT> DFadType;
    typedef Intrepid::Tensor<ScalarT> Tensor;

    ///
    /// Local functions in evaluating GursonFD model stress:
    ///
    ScalarT
    YieldFunction(Tensor const & s, ScalarT const & p, ScalarT const & fvoid,
        ScalarT const & eq, ScalarT const & K, ScalarT const & Y,
        ScalarT const & siginf,ScalarT const & delta,
        ScalarT const & Jacobian, ScalarT const & E);

    void
    ResidualJacobian(std::vector<ScalarT> & X,
      std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,  const ScalarT & p,
      const ScalarT & fvoid, const ScalarT & eq, Tensor & s,
      const ScalarT & shearModulus, const ScalarT & bulkModulus,
      const ScalarT & K, const ScalarT & Y, const ScalarT & siginf,
      const ScalarT & delta, const ScalarT & Jacobian);

    ///
    /// Input: Deformation Gradient
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;

    ///
    /// Input: Determinant of Deformation Gradient
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> J;

    ///
    /// Input: Young's Modulus
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;

    ///
    /// Input: Poisson's Ratio
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

    ///
    /// Input: Yield Strength
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;

    ///
    /// Input: Hardening Modulus
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;

    ///
    /// Input: Saturation Modulus
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> saturationModulus;

    ///
    /// Input: Saturation Exponent
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> saturationExponent;

    ///
    /// Input: Time Increment
    ///
    PHX::MDField<ScalarT,Dummy> deltaTime;

    ///
    /// Number of Integration Points
    ///
    unsigned int numQPs;

    ///
    /// Number of Global Dimensions
    ///
    unsigned int numDims;

    ///
    /// Number of Elements in the Workset
    ///
    unsigned int worksetSize;

    ///
    /// State Variable Name
    ///
    std::string fpName, eqpsName, voidVolumeName, defGradName, stressName;

    ///
    /// Input: Constant Material Properties
    ///
    RealType eq0;
    RealType N;
    RealType f0;
    RealType kw;
    RealType eN;
    RealType sN;
    RealType fN;
    RealType fc;
    RealType ff;
    RealType q1;
    RealType q2;
    RealType q3;

    ///
    /// Input: Flags for Saturation Hardening
    ///
    bool isSaturationH;

    ///
    /// ParameterList storing material constants
    ///
    Teuchos::ParameterList* pList;


    ///
    /// Output: Cauchy Stress
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;

    ///
    /// Output: Plastic Deformation Gradient
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;

    ///
    /// Output: Equivalent Plastic Strain
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;

    ///
    /// Output: Void Volume Fraction
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> voidVolume;

    Intrepid::FieldContainer<ScalarT> Fpinv;
    Intrepid::FieldContainer<ScalarT> FpinvT;
    Intrepid::FieldContainer<ScalarT> Cpinv;

  };
}

#endif

