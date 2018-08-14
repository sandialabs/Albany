//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HDIFFUSIONDEFORMATION_MATTER_RESIDUAL_HPP
#define HDIFFUSIONDEFORMATION_MATTER_RESIDUAL_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief

    This evaluator computes the residue of the hydrogen concentration
    equilibrium equation.

*/

template <typename EvalT, typename Traits>
class HDiffusionDeformationMatterResidual
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  HDiffusionDeformationMatterResidual(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>            weights;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> GradBF;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      DefGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                elementLength;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                Dstar;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                DL;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                Clattice;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>           CLGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>           stressGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                stabParameter;

  // Input for the strain rate effect
  PHX::MDField<const ScalarT, Cell, QuadPoint> Ctrapped;
  PHX::MDField<const ScalarT, Cell, QuadPoint> Ntrapped;
  PHX::MDField<const ScalarT, Cell, QuadPoint> eqps;
  PHX::MDField<const ScalarT, Cell, QuadPoint> eqpsFactor;
  std::string                                  eqpsName;

  // Input for hydro-static stress effect
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> Pstress;
  PHX::MDField<const ScalarT, Cell, QuadPoint>           tauFactor;

  // Time
  PHX::MDField<const ScalarT, Dummy> deltaTime;

  // Data from previous time step
  std::string ClatticeName;
  std::string CLGradName;

  // bool haveSource;
  // bool haveMechSource;
  bool enableTransient;

  bool have_eqps_;

  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // Temporary Views
  Kokkos::DynRankView<ScalarT, PHX::Device> Hflux;
  Kokkos::DynRankView<ScalarT, PHX::Device> C;
  Kokkos::DynRankView<ScalarT, PHX::Device> Cinv;
  Kokkos::DynRankView<ScalarT, PHX::Device> CinvTgrad;
  Kokkos::DynRankView<ScalarT, PHX::Device> CinvTgrad_old;
  Kokkos::DynRankView<ScalarT, PHX::Device> artificalDL;
  Kokkos::DynRankView<ScalarT, PHX::Device> stabilizedDL;
  Kokkos::DynRankView<ScalarT, PHX::Device> pterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> tpterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> CinvTaugrad;

  ScalarT CLbar, vol;
  ScalarT trialPbar;

  RealType stab_param_, t_decay_constant_;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;
};
}  // namespace LCM

#endif
