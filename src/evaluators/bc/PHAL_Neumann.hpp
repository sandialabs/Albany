//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
#include "PHAL_Utilities.hpp"

#include "Albany_MaterialDatabase.hpp"


namespace PHAL {

/** \brief Neumann boundary condition evaluator

*/


template<typename EvalT, typename Traits>
class NeumannBase :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  enum NEU_TYPE {COORD, NORMAL, INTJUMP, PRESS, ROBIN, TRACTION, CLOSED_FORM, STEFAN_BOLTZMANN};
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

  using ICT = Intrepid2::CellTools<PHX::Device>;
  using IRST = Intrepid2::RealSpaceTools<PHX::Device>;
  using IFST = Intrepid2::FunctionSpaceTools<PHX::Device>;

  const Teuchos::RCP<Albany::Layouts>& dl;
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;

  int  cellDims,  numQPs, numNodes, numCells, maxSideDim, maxNumQpSide;
  Teuchos::Array<int> offset;
  int numDOFsSet;

  // Should only specify flux vector components (dudx, dudy, dudz), dudn, or pressure P

  // dudn scaled
  void calc_dudn_const(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                          ScalarT scale = 1.0) const ;

  // robin (also uses flux scaling)
  void calc_dudn_robin (Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                        const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const;

  // Stefan-Boltzmann (also uses flux scaling)
  void calc_dudn_radiate (Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                        const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const;

   // (dudx, dudy, dudz)
  void calc_gradu_dotn_const(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                          const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          int local_side_id) const;

   // (t_x, t_y, t_z)
  void calc_traction_components(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned) const ;

   // Pressure P
  void calc_press(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                          const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          int local_side_id) const;

  // closed_from bc assignment
  void calc_closed_form(Kokkos::DynRankView<ScalarT, PHX::Device> &    qp_data_returned,
                        const Kokkos::DynRankView<MeshScalarT, PHX::Device>& physPointsSide,
                        const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                        const shards::CellTopology & celltopo,
                        int local_side_id,
                        typename Traits::EvalData workset) const;

   // Do the side integration
  void evaluateNeumannContribution(typename Traits::EvalData d);

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<const MeshScalarT,Cell,Vertex,Dim> coordVec;
  PHX::MDField<const ScalarT> dof;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::ArrayRCP<Teuchos::RCP<shards::CellTopology> > sideType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubatureCell;
  Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Temporary Views
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell_buffer;

  Kokkos::DynRankView<ScalarT, PHX::Device> dofCell_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofCellVec_buffer;


  Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> refPointsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide_buffer;
  Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide_buffer;

  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_det_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_trans_basis_refPointsSide_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> side_normals_buffer;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> normal_lengths_buffer;

  Kokkos::DynRankView<ScalarT, PHX::Device> dofSide_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofSideVec_buffer;

  Kokkos::DynRankView<MeshScalarT, PHX::Device> temporary_buffer;
  Kokkos::DynRankView<ScalarT, PHX::Device> data_buffer;

  // Output:
  Kokkos::DynRankView<ScalarT, PHX::Device> neumann;

  int numSidesOnElem;
  bool vectorDOF;

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

  MDFieldMemoizer<Traits> memoizer;
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
 Teuchos::RCP<Tpetra_CrsMatrix> JacT;

 typedef typename Tpetra_CrsMatrix::local_matrix_type  LocalMatrixType;
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

  void postRegistrationSetup(typename Traits::SetupData /* d */,
                             PHX::FieldManager<Traits>& /* vm */) {};

  void evaluateFields(typename Traits::EvalData /* d */) {};

};

}

#endif
