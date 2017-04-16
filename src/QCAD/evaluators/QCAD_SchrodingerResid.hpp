//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_SCHRODINGERRESID_HPP
#define QCAD_SCHRODINGERRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Albany_MaterialDatabase.hpp"

namespace QCAD {

template<typename EvalT, typename Traits>
class SchrodingerResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  SchrodingerResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  //! Helper function to compute inverse effective mass
  double getInvEffMassFiniteWall( const PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> & coord,
                                  const int cell, const int qp );
  double getInvEffMass1DMosCap(const MeshScalarT coord0);
  
  // Input:
  std::size_t numCells;
  std::size_t numQPs;
  std::size_t numDims;

  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint> psi;
  PHX::MDField<const ScalarT,Cell,QuadPoint> psiDot;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> psiGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint> V;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  
  bool enableTransient;
  bool havePotential;
  bool bOnlyInQuantumBlocks;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> psiResidual;

  // Intermediate workspace
  Kokkos::DynRankView<ScalarT, PHX::Device> psiGradWithMass;
  Kokkos::DynRankView<ScalarT, PHX::Device> psiV;
  Kokkos::DynRankView<ScalarT, PHX::Device> V_barrier;

  //! units
  double energy_unit_in_eV, length_unit_in_m;
  
  //! Material database
  Teuchos::RCP<Albany::MaterialDatabase> materialDB;
  
  //! Parameters for Finite Wall potential
  std::string potentialType;
  double barrEffMass; // in [m0]
  double barrWidth;   // in length_unit_in_m
  double wellEffMass;
  double wellWidth; 

  //! Parameters for 1D MOSCapacitor to set effective mass for oxide and silicon regions
  double oxideWidth;
  double siliconWidth;
  
  double hbar2_over_2m0;  // in energy_unit_in_eV
    
};

}

#endif
