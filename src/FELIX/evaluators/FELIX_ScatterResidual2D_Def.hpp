//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
ScatterResidual2D<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
    numFields(ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{

  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");
  if (p.isType<int>("Field Level"))
    fieldLevel = p.get<int>("Field Level");
  else fieldLevel = -1;
  if (p.isType<const std::string>("Mesh Part"))
    meshPart = p.get<const std::string>("Mesh Part");
  else
    meshPart = "upperside";
}

// **********************************************************************
template<typename Traits>
void ScatterResidual2D<PHAL::AlbanyTraits::Jacobian, Traits>::
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
  fieldLevel = (fieldLevel < 0) ? numLayers : fieldLevel;
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
          if(il_col != fieldLevel) {
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
ScatterResidual2D<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
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
ScatterResidual2D<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl)
{
}

#endif


// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
    numFields(ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
  if (p.isType<int>("Offset 2D Field"))
    offset2DField = p.get<int>("Offset 2D Field");
  else offset2DField = numFields-1;
  if (p.isType<int>("Field Level"))
    fieldLevel = p.get<int>("Field Level");
  else fieldLevel = -1;
}

// **********************************************************************
template<typename Traits>
void ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;
  const bool loadResid = Teuchos::nonnull(fT);
  const int neq = workset.wsElNodeEqID[0][0].size();
  unsigned int nunk = this->numNodes*(neq-1);
  Teuchos::Array<LO> colT, index;
  colT.resize(nunk), index.resize(nunk);
  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    // Local Unks: Loop over nodes in element, Loop over equations per node
    for (unsigned int node_col(0), i(0); node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        if(eq_col != offset2DField) {
          colT[i] = nodeID[node_col][eq_col];
          index[i++] = neq * node_col + eq_col;
        }
      }
    }
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if(this->offset + eq != offset2DField) {
          typename PHAL::Ref<ScalarT>::type
            valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                      this->valTensor[0](cell,node, eq/numDim, eq%numDim));
          const LO rowT = nodeID[node][this->offset + eq];
          if (loadResid)
            fT->sumIntoLocalValue(rowT, valptr.val());
          // Check derivative array is nonzero
          if (valptr.hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int lunk = 0; lunk < nunk; lunk++) {
                JacT->sumIntoLocalValues( colT[lunk], Teuchos::arrayView(&rowT, 1),
                  Teuchos::arrayView(&(valptr.fastAccessDx(index[lunk])), 1));
              }

            }
            else {
              double two=2;
              // Sum Jacobian entries all at once
              for (unsigned int lunk = 0; lunk < nunk; lunk++) {
                JacT->sumIntoLocalValues( rowT, Teuchos::arrayView(&colT[lunk],1),
                   Teuchos::arrayView(&(valptr.fastAccessDx(index[lunk])), 1));
              }
            }
          } // has fast access
        }
      }
    }
  }

  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  fieldLevel = (fieldLevel < 0) ? numLayers : fieldLevel;
  colT.resize(this->numNodes);

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    Teuchos::ArrayRCP<LO> basalIds(this->numNodes);
    LO base_id, ilayer;
    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node_col]);
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
      colT[node_col] = solDOFManager.getLocalDOF(inode, offset2DField);
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        const LO rowT = nodeID[node][eq];
        if(eq != offset2DField) {
          typename PHAL::Ref<ScalarT>::type valptr = this->valVec(cell,node,eq);

          if (loadResid)
            fT->sumIntoLocalValue(rowT, valptr.val());

          if (valptr.hasFastAccess()) { // has fast access
              // Sum Jacobian entries all at once
            for (unsigned int i = 0; i < this->numNodes; i++)
              JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&colT[i],1), Teuchos::arrayView(&(valptr.fastAccessDx(neq*i + offset2DField)), 1));
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
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl)
{
}


// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
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
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{
}



// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidual<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl)
{
}


#endif
}

