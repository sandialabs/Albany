//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_AnisotropicHyperelasticDamage_hpp)
#define LCM_AnisotropicHyperelasticDamage_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp" 
#include "Tensor.h"

namespace LCM {
  ///\brief Hyperelasticity with 2 Fiber families and damage
  ///
  /// Anisotropic damage in a hyperelastic context
  ///
  template<typename EvalT, typename Traits>
  class AnisotropicHyperelasticDamage : public PHX::EvaluatorWithBaseImpl<Traits>,
                                        public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    AnisotropicHyperelasticDamage(const Teuchos::ParameterList& p,
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

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

  private:

    ///
    /// Input: Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;

    ///
    /// Input: Determinant of Deformation Gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> J;

    ///
    /// Input: Young's Modulus
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;

    ///
    /// Input: Poisson's Ratio
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

    ///
    /// Input: Coordinates of integration points
    ///
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

    ///
    /// ParameterList storing material constants
    ///
    Teuchos::ParameterList* pList;

    ///
    /// Output: Cauchy Stress
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

    ///
    /// Output: Matrix Energy
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> energyM;

    ///
    /// Output: Fiber 1 Energy
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> energyF1;

    ///
    /// Output: Fiber 2 Energy
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> energyF2;

    ///
    /// Output: Matrix Damage
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> damageM;

    ///
    /// Output: Fiber 1 Damage
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> damageF1;

    ///
    /// Output: Fiber 2 Damage
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> damageF2;

    ///
    /// Name of the Matrix Energy
    ///
    std::string energyMName;

    ///
    /// Name of the Fiber 1 Energy
    ///
    std::string energyF1Name;

    ///
    /// Name of the Fiber 1 Energy
    ///
    std::string energyF2Name;

    ///
    /// numver of integration points
    ///
    unsigned int numQPs;

    ///
    /// number of global dimensions
    ///
    unsigned int numDims;

    ///
    /// number of elements in the workset
    ///
    unsigned int worksetSize;

    ///
    /// Material parameters for Fiber 1
    ///
    RealType kF1, qF1, volFracF1, damageMaxF1, saturationF1;

    ///
    /// Material parameters for Fiber 2
    ///
    RealType kF2, qF2, volFracF2, damageMaxF2, saturationF2;

    ///
    /// Material parameters for Matrix
    ///
    RealType volFracM, damageMaxM, saturationM;

    std::vector< RealType > directionF1;
    std::vector< RealType > directionF2;
    std::vector< RealType > ringCenter;
    bool isLocalCoord;
  };
}

#endif
