//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HDIFFUSIONDEFORMATION_MATTER_RESIDUAL_HPP
#define HDIFFUSIONDEFORMATION_MATTER_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief

    This evaluator computes the residue of the hydrogen concentration
    equilibrium equation.

*/

template<typename EvalT, typename Traits>
class HDiffusionDeformationMatterResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
				public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  HDiffusionDeformationMatterResidual(Teuchos::ParameterList& p,
                                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> Source;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> DefGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> elementLength;
  PHX::MDField<ScalarT,Cell,QuadPoint> Dstar;
  PHX::MDField<ScalarT,Cell,QuadPoint> DL;
  PHX::MDField<ScalarT,Cell,QuadPoint> Clattice;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> CLGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> stressGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> stabParameter;

  // Input for the strain rate effect
  PHX::MDField<ScalarT,Cell,QuadPoint> Ctrapped;
  PHX::MDField<ScalarT,Cell,QuadPoint> Ntrapped;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps;
  PHX::MDField<ScalarT,Cell,QuadPoint> eqpsFactor;
  std::string eqpsName;


  // Input for hydro-static stress effect
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Pstress;
  PHX::MDField<ScalarT,Cell,QuadPoint> tauFactor;

  // Time
  PHX::MDField<ScalarT,Dummy> deltaTime;

  //Data from previous time step
  std::string ClatticeName;
  std::string CLGradName;

  //bool haveSource;
  //bool haveMechSource;
  bool enableTransient;

  bool have_eqps_;
  
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> Hflux;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> C;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> Cinv;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> CinvTgrad;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> CinvTgrad_old;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> artificalDL;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> stabilizedDL;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> tauStress;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> pterm;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> tpterm;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> tauH;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> CinvTaugrad;


  ScalarT CLbar, vol ;
  ScalarT trialPbar;

  RealType stab_param_, t_decay_constant_;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> TResidual;


};
}

#endif
