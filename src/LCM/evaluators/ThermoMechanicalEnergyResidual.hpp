//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMO_MECHANICAL_ENERGY_RESIDUAL_HPP
#define THERMO_MECHANICAL_ENERGY_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
  /** \brief ThermMechanical Energy Residual

      This evaluator computes the residual for the energy equation
      in the coupled therm-mechanical problem

  */

  template<typename EvalT, typename Traits>
  class ThermoMechanicalEnergyResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
					 public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ThermoMechanicalEnergyResidual(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

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
    PHX::MDField<ScalarT,Dummy> deltaTime; // time step
    RealType density;
    RealType Cv;

    // Output:
    PHX::MDField<ScalarT,Cell,Node> TResidual;

    bool haveSource;
    std::string tempName;
    unsigned int numQPs, numDims, worksetSize;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> flux;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> C;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> Cinv;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> CinvTgrad;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> Tdot;
  };
}

#endif
