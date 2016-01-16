//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef QCAD_POISSONSOURCEINTERFACE_HPP
#define QCAD_POISSONSOURCEINTERFACE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include "Albany_ProblemUtils.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "QCAD_MaterialDatabase.hpp"


namespace QCAD {

/** \brief Include interface traps into the Poisson equation

*/


template<typename EvalT, typename Traits>
class PoissonSourceInterfaceBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  enum SIDE_TYPE {OTHER, LINE, TRI, QUAD}; // to calculate areas for pressure bc

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // PoissonSourceInterfaceBase(const Teuchos::ParameterList& p);
  PoissonSourceInterfaceBase(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d) = 0;

  //! Public Universal Constants
  /***** define universal constants as double constants *****/
  static const double kbBoltz; // Boltzmann constant in [eV/K]
  static const double eps0; // vacuum permittivity in [C/(V.cm)]
  static const double eleQ; // electron elemental charge in [C]
  static const double m0;   // vacuum electron mass in [kg]
  static const double hbar; // reduced planck constant in [J.s]
  static const double pi;   // pi constant (unitless)
  static const double MAX_EXPONENT;   

protected:

  const Teuchos::RCP<Albany::Layouts>& dl;
  const Teuchos::RCP<const Albany::MeshSpecsStruct>& meshSpecs;

  int  cellDims,  numQPs, numNodes;
  int numDOFsSet;

  double energy_unit_in_eV;
  double length_unit_in_meters, X0; 
  double temperature, kbT;

  std::vector<double> elecAffinity, bandGap, fermiEnergy;
  std::vector<double> trapDensity, acceptorDegFac, donorDegFac; 
  std::vector<std::string> trapSpectrum; 

  Teuchos::Array<int> offset;
  Teuchos::RCP<MaterialDatabase> materialDB;

  bool responseOnly; //flag for evaluator being called in response field manager
  
  ScalarT qPhiRef; 

/*
  // dudn for 2D Thomas-Fermi poisson source
  void calc_dudn_2DThomasFermi(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
			       const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
			       const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side,
			       const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
			       const shards::CellTopology & celltopo,
			       const int cellDims,
			       int local_side_id, int iSideset);
*/

  ScalarT getReferencePotential();

  // Compute -q*X0/eps0*Nit, where Nit is the interface trap charge density in [#/cm^2]
  void calcInterfaceTrapChargDensity(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
			       const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side, int iSideset); 

   // Perform the finite element integration over an interface (2D for now, currently not support 1D)
  void evaluateInterfaceContribution(typename Traits::EvalData d);

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,Node> dof;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::ArrayRCP<Teuchos::RCP<shards::CellTopology> > sideType;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureCell;
  Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubPointsSide;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPointsSide;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubWeightsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide_det;

  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsCell;

  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_measure;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_trans_basis_refPointsSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCell;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> data;

  // Output:
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>   neumann;

  Teuchos::Array<std::string> sideSetIDs;
  //Teuchos::Array<RealType> inputValues;
  //std::string inputConditions;
  //std::string name;

  Teuchos::Array<SIDE_TYPE> side_type;
};

template<typename EvalT, typename Traits> class PoissonSourceInterface;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::Residual,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::Jacobian,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::Tangent,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::SGResidual,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::SGTangent,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::MPResidual,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class PoissonSourceInterface<PHAL::AlbanyTraits::MPTangent,Traits>
  : public PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  PoissonSourceInterface(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif



}

#endif
