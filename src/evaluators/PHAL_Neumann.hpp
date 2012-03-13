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

namespace PHAL {

/** \brief Neumann boundary condition evaluator

*/

template<typename EvalT, typename Traits>
class Neumann : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  Neumann(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  cellDims, sideDims, numQPs, numQPsSide, numNodes;

  void calc_dTdn_const(Intrepid::FieldContainer<MeshScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

  void calc_gradT_dotn_const(Intrepid::FieldContainer<MeshScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id);

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

  Intrepid::FieldContainer<MeshScalarT> weighted_measure;
  Intrepid::FieldContainer<RealType> basis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> trans_basis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> weighted_trans_basis_refPointsSide;

  Intrepid::FieldContainer<MeshScalarT> data;

  // Output:
  PHX::MDField<MeshScalarT,Cell,Node>   neumann;

  std::string sideSetID;

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
  
  NeumannAggregator(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {};
  
  void evaluateFields(typename Traits::EvalData d) {};

};

}

#endif
