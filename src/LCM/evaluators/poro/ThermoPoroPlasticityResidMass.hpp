//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMOPOROPLASTICITYRESIDMASS_HPP
#define THERMOPOROPLASTICITYRESIDMASS_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief
 *   Balance of energy residual for large deformation
 *   thermoporomechanics problem.

*/

template <typename EvalT, typename Traits>
class ThermoPoroPlasticityResidMass
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ThermoPoroPlasticityResidMass(const Teuchos::ParameterList& p);

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
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                porePressure;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                densityPoreFluid;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                Temp;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                RefTemp;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                stabParameter;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                ThermalCond;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                kcPermeability;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                porosity;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                biotCoefficient;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                biotModulus;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                young_modulus_;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                poissons_ratio_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>           TGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim>           TempGrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                alphaMixture;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                alphaPoreFluid;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                alphaSkeleton;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                Source;
  Teuchos::Array<double>                                      convectionVels;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                rhoCp;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                Absorption;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      strain;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      defgrad;
  PHX::MDField<const ScalarT, Cell, QuadPoint>                J;

  // stabilization term
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim> coordVec;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>     cubature;
  Teuchos::RCP<shards::CellTopology>                 cellType;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint>   weights;

  // Time
  PHX::MDField<const ScalarT, Dummy> deltaTime;

  // Data from previous time step
  std::string strainName, porePressureName, porosityName, JName, TempName;

  bool         haveSource;
  bool         haveConvection;
  bool         haveAbsorption;
  bool         enableTransient;
  bool         haverhoCp;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // Temporary Views
  Kokkos::DynRankView<ScalarT, PHX::Device> flux;
  Kokkos::DynRankView<ScalarT, PHX::Device> fgravity;
  Kokkos::DynRankView<ScalarT, PHX::Device> fluxdt;
  Kokkos::DynRankView<ScalarT, PHX::Device> pterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> Tterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> aterm;
  Kokkos::DynRankView<ScalarT, PHX::Device> tpterm;

  // Work space FCs
  Kokkos::DynRankView<ScalarT, PHX::Device> F_inv;
  Kokkos::DynRankView<ScalarT, PHX::Device> F_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> C;
  Kokkos::DynRankView<ScalarT, PHX::Device> Cinv;
  Kokkos::DynRankView<ScalarT, PHX::Device> JF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> KJF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> Kref;

  ScalarT porePbar, Tempbar, vol;
  ScalarT trialPbar;
  ScalarT shearModulus, bulkModulus;
  ScalarT safeFactor;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> TResidual;
};
}  // namespace LCM

#endif
