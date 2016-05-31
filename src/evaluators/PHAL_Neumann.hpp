//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_NEUMANN_HPP
#define PHAL_NEUMANN_HPP

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


namespace PHAL {

/** \brief Neumann boundary condition evaluator

*/


template<typename EvalT, typename Traits>
class NeumannBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  enum NEU_TYPE {COORD, NORMAL, INTJUMP, PRESS, ROBIN, BASAL, BASAL_SCALAR_FIELD, TRACTION, LATERAL};
  enum SIDE_TYPE {OTHER, LINE, TRI, QUAD}; // to calculate areas for pressure bc

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  NeumannBase(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d) = 0;

  ScalarT& getValue(const std::string &n);

protected:

  const Teuchos::RCP<Albany::Layouts>& dl;
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;

  int  cellDims,  numQPs, numNodes;
  Teuchos::Array<int> offset;
  int numDOFsSet;

  //The following are for the basal BC 
  std::string betaName; //name of function betaXY to be used
  double L;           //length scale for ISMIP-HOM Test cases 
  MeshScalarT betaXY; //function of x and y to multiply scalar values of beta read from input file
  enum BETAXY_NAME {CONSTANT, EXPTRIG, ISMIP_HOM_TEST_C, ISMIP_HOM_TEST_D, CONFINEDSHELF, CIRCULARSHELF, DOMEUQ, SCALAR_FIELD, EXP_SCALAR_FIELD, POWERLAW_SCALAR_FIELD, EXP_SCALAR_FIELD_THK, GLP_SCALAR_FIELD, LATERAL_BACKPRESSURE, FELIX_XZ_MMS};
  BETAXY_NAME beta_type;
 
  //The following are for the lateral BC 
  double g; 
  double rho; 
  double rho_w;
  Teuchos::ParameterList* stereographicMapList;
  bool useStereographicMap;

 // Should only specify flux vector components (dudx, dudy, dudz), dudn, or pressure P

   // dudn scaled
  void calc_dudn_const(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id,
                          ScalarT scale = 1.0);

  // robin (also uses flux scaling)
  void calc_dudn_robin (Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                        const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
                        const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side,
                        const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                        const shards::CellTopology & celltopo,
                        const int cellDims,
                        int local_side_id,
                        ScalarT scale,
                        const ScalarT* robin_param_values);

   // (dudx, dudy, dudz)
  void calc_gradu_dotn_const(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

   // (t_x, t_y, t_z)
  void calc_traction_components(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

   // Pressure P
  void calc_press(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& phys_side_cub_points,
                          const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

  //Basal bc
  void calc_dudn_basal(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
   		       const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& basalFriction_side,
   		       const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& thickness_side,
   		      const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& bedTopo_side,
   		       const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side,
                       const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                       const shards::CellTopology & celltopo,
                       const int cellDims,
                       int local_side_id);

  void calc_dudn_basal_scalar_field(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                            const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& basalFriction_side,
                            const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side,
                                const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                                const shards::CellTopology & celltopo,
                                const int cellDims,
                                int local_side_id);

  //Lateral bc
  void calc_dudn_lateral(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
                     const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& thickness_side,
                     const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& elevation_side,
                     const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side,
                         const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>& jacobian_side_refcell,
                         const shards::CellTopology & celltopo,
                         const int cellDims,
                         int local_side_id);

   // Do the side integration
  void evaluateNeumannContribution(typename Traits::EvalData d);

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,Node> dof;
  PHX::MDField<ScalarT,Cell,Node,VecDim> dofVec;
  PHX::MDField<ParamScalarT,Cell,Node> beta_field;
  PHX::MDField<ParamScalarT,Cell,Node> roughness_field;
  PHX::MDField<ParamScalarT,Cell,Node> thickness_field;
  PHX::MDField<ParamScalarT,Cell,Node> elevation_field;
  PHX::MDField<ParamScalarT,Cell,Node> bedTopo_field;
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

  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_measure;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_trans_basis_refPointsSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCell;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCellVec;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSideVec;
  
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> data;

  // Output:
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>   neumann;

  std::string sideSetID;
  Teuchos::Array<RealType> inputValues;
  std::string inputConditions;
  std::string name;

  NEU_TYPE bc_type;
  Teuchos::Array<SIDE_TYPE> side_type;
  ScalarT const_val;
  ScalarT robin_vals[5]; // (dof_value, coeff multiplying difference (dof - dof_value), jump)
  std::vector<ScalarT> dudx;

  std::vector<ScalarT> matScaling;

};

template<typename EvalT, typename Traits> class Neumann;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::Residual,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::Jacobian,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

 Teuchos::RCP<Tpetra_Vector> fT;
 Teuchos::ArrayRCP<ST> fT_nonconstView;
 Teuchos::RCP<Tpetra_CrsMatrix> JacT;

 typedef typename Tpetra_CrsMatrix::k_local_matrix_type  LocalMatrixType;
 LocalMatrixType jacobian;
 Kokkos::View<int***, PHX::Device> Index;
 bool is_adjoint;

 typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

 struct Neumann_Tag{};
 typedef Kokkos::RangePolicy<ExecutionSpace, Neumann_Tag> Neumann_Policy;

 KOKKOS_INLINE_FUNCTION
  void operator() (const Neumann_Tag& tag, const int& i) const;

#endif

};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::Tangent,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::SGResidual,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::SGTangent,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
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
class Neumann<PHAL::AlbanyTraits::MPResidual,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::MPTangent,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  Neumann(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif


// **************************************************************
// **************************************************************
// Evaluator to aggregate all Neumann BCs into one "field"
// **************************************************************
template<typename EvalT, typename Traits>
class NeumannAggregator
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{
private:

  typedef typename EvalT::ScalarT ScalarT;

public:
  
  NeumannAggregator(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {};
  
  void evaluateFields(typename Traits::EvalData d) {};

};

}

#endif
