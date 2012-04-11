/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#ifndef PHAL_NEUMANN_HPP
#define PHAL_NEUMANN_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

#include "Albany_ProblemUtils.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "QCAD_MaterialDatabase.hpp"


namespace PHAL {

/** \brief Neumann boundary condition evaluator

*/

enum NEU_TYPE {COORD, NORMAL, INTJUMP};

template<typename EvalT, typename Traits>
class NeumannBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits>
   {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  NeumannBase(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d) = 0;

  ScalarT& getValue(const std::string &n);


protected:

  const Teuchos::RCP<Albany::Layouts>& dl;
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;

  int  cellDims, sideDims, numQPs, numQPsSide, numNodes;
  const int offset;

 // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both

   // dudn scaled
  void calc_dudn_const(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id,
                          ScalarT scale = 1.0);

   // (dudx, dudy, dudz)
  void calc_gradu_dotn_const(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

   // Do the side integration
  void evaluateNeumannContribution(typename Traits::EvalData d);

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureCell;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> cubPointsSide;
  Intrepid::FieldContainer<RealType> refPointsSide;
  Intrepid::FieldContainer<RealType> cubWeightsSide;
  Intrepid::FieldContainer<MeshScalarT> physPointsSide;
  Intrepid::FieldContainer<MeshScalarT> jacobianSide;
  Intrepid::FieldContainer<MeshScalarT> jacobianSide_det;

  Intrepid::FieldContainer<MeshScalarT> physPointsCell;

  Intrepid::FieldContainer<MeshScalarT> weighted_measure;
  Intrepid::FieldContainer<RealType> basis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> trans_basis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> weighted_trans_basis_refPointsSide;

  Intrepid::FieldContainer<ScalarT> data;

  // Output:
//  PHX::MDField<MeshScalarT,Cell,Node>   neumann;
//  Intrepid::FieldContainer<MeshScalarT>   neumann;
  Intrepid::FieldContainer<ScalarT>   neumann;

  std::string sideSetID;
  Teuchos::Array<RealType> inputValues;
  std::string inputConditions;
  std::string name;

  NEU_TYPE bc_type;
  ScalarT dudn;
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
  Neumann(const Teuchos::ParameterList& p);
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
  Neumann(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::Tangent,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  Neumann(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::SGResidual,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  Neumann(const Teuchos::ParameterList& p);
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
  Neumann(const Teuchos::ParameterList& p);
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
  Neumann(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class Neumann<PHAL::AlbanyTraits::MPResidual,Traits>
  : public NeumannBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  Neumann(const Teuchos::ParameterList& p);
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
  Neumann(const Teuchos::ParameterList& p);
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
  Neumann(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
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
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {};
  
  void evaluateFields(typename Traits::EvalData d) {};

};

}

#endif
