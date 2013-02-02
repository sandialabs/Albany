//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_EnergyResidual_hpp)
#define LCM_EnergyResidual_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  ///
  ///\brief Residual for Balance of Energy
  ///
  template<typename EvalT, typename Traits>
  class EnergyResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    EnergyResidual(const Teuchos::ParameterList& p,
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

    // Input:
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
    PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;
    PHX::MDField<ScalarT,Cell,QuadPoint> ThermalCond;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> TGrad;
    PHX::MDField<ScalarT,Cell,QuadPoint> Source;
    PHX::MDField<ScalarT,Cell,QuadPoint> Absorption;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F; // deformation gradient
    PHX::MDField<ScalarT,Cell,QuadPoint> mechSource; // mechanical heat source
    PHX::MDField<ScalarT,Cell,Dummy> deltaTime; // time step
    RealType density;
    RealType Cv;

    // Output:
    PHX::MDField<ScalarT,Cell,Node> TResidual;

    bool haveSource;
    std::string tempName;
    unsigned int numQPs, numDims, worksetSize;
    Intrepid::FieldContainer<ScalarT> flux;
    Intrepid::FieldContainer<ScalarT> C;
    Intrepid::FieldContainer<ScalarT> Cinv;
    Intrepid::FieldContainer<ScalarT> CinvTgrad;
    Intrepid::FieldContainer<ScalarT> Tdot;
  };
}

#endif
