//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
    numFields(ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{

  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");
  if (p.isType<int>("H level"))
    HLevel = p.get<int>("H level");
  else HLevel = -1;
  if (p.isType<const std::string>("Mesh Part"))
    meshPart = p.get<const std::string>("Mesh Part");
  else
    meshPart = "upperside";
}

// **********************************************************************
template<typename Traits>
void ScatterResidualH<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;
  const bool loadResid = Teuchos::nonnull(fT);
  Teuchos::Array<LO> colT;
  const int neq = workset.wsElNodeEqID[0][0].size();
  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor[0].dimension(2);
  double diagonal_value = 1;

  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  HLevel = (HLevel < 0) ? numLayers : HLevel;
  colT.reserve(neq*this->numNodes*(numLayers+1));


  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets not properly specified on the mesh" << std::endl);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(meshPart);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition

    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const int elem_GID = sideSet[iSide].elem_GID;
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      int numSideNodes = side.topology->node_count;

      colT.resize(neq*numSideNodes*(numLayers+1));
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[elem_LID];

      LO base_id, ilayer;
      for (int i = 0; i < numSideNodes; ++i) {
        std::size_t node = side.node[i];
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        for (unsigned int il_col=0; il_col<numLayers+1; il_col++) {
          LO inode = layeredMeshNumbering.getId(base_id, il_col);
          for (unsigned int eq_col=0; eq_col<neq; eq_col++)
            colT[il_col*neq*numSideNodes + neq*i + eq_col] = solDOFManager.getLocalDOF(inode, eq_col);
          if(il_col != HLevel) {
            const LO rowT = solDOFManager.getLocalDOF(inode, this->offset); //insert diagonal values
            JacT->replaceLocalValues(rowT, Teuchos::Array<LO>(1, rowT), Teuchos::arrayView(&diagonal_value, 1));
          }
        }
      }

      for (int i = 0; i < numSideNodes; ++i) {
        std::size_t node = side.node[i];
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](elem_LID,node) :
                    this->tensorRank == 1 ? this->valVec(elem_LID,node,eq) :
                    this->valTensor[0](elem_LID,node, eq/numDim, eq%numDim));
          const LO rowT = nodeID[node][this->offset + eq];
          if (loadResid)
            fT->sumIntoLocalValue(rowT, valptr.val());
          if (valptr.hasFastAccess()) {
            JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)),colT.size()));
          } // has fast access
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGTangent,Traits>(p,dl)
{
}
#endif

#ifdef ALBANY_ENSEMBLE
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl)
{
}

#endif


// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
    numFields(ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
  if (p.isType<int>("H Offset"))
    HOffset = p.get<int>("H Offset");
  else HOffset = 2;
  if (p.isType<int>("H level"))
    HLevel = p.get<int>("H level");
  else HLevel = -1;
}

// **********************************************************************
template<typename Traits>
void ScatterResidualH3D<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;
  const bool loadResid = Teuchos::nonnull(fT);
  Teuchos::Array<LO> colT;
  colT.resize(this->numNodes);
  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor[0].dimension(2);

  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  HLevel = (HLevel < 0) ? numLayers : HLevel;

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    Teuchos::ArrayRCP<LO> basalIds(this->numNodes);
    LO base_id, ilayer;
    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node_col]);
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      LO inode = layeredMeshNumbering.getId(base_id, HLevel);
      colT[node_col] = solDOFManager.getLocalDOF(inode, HOffset);
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        const LO rowT = nodeID[node][eq];
        if(eq != HOffset) {
          typename PHAL::Ref<ScalarT>::type valptr = this->valVec(cell,node,eq);

          if (loadResid)
            fT->sumIntoLocalValue(rowT, valptr.val());

          if (valptr.hasFastAccess()) { // has fast access
              // Sum Jacobian entries all at once
              JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), this->numNodes));
          }
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGTangent,Traits>(p,dl)
{
}
#endif

#ifdef ALBANY_ENSEMBLE
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualH3D<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl)
{
}


#endif
}

