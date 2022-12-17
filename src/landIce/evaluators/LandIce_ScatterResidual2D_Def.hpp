//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_NodalDOFManager.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "LandIce_ScatterResidual2D.hpp"

namespace PHAL {

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{
  // Nothing to do here
}

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
  fieldLevel = p.get<int>("Field Level");
  meshPart = p.get<std::string>("Mesh Part");
}

// **********************************************************************
template<typename Traits>
void ScatterResidual2D<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  const bool loadResid = Teuchos::nonnull(workset.f);
  Teuchos::Array<LO> lcols;
  const unsigned int neq = nodeID.extent(2);
  unsigned int numDim = 0;
  if (this->tensorRank==2) {
    numDim = this->valTensor.extent(2);
  }
  double diagonal_value = 1;

  if (workset.sideSets == Teuchos::null) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets not properly specified on the mesh" << std::endl);
  }

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(meshPart);

  auto Jac = workset.Jac;
  Teuchos::ArrayRCP<ST> f_data;
  if (loadResid) {
    f_data = Albany::getNonconstLocalData(workset.f);
  }

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
    auto solIndexer = workset.disc->getOverlapGlobalLocalIndexer();
    const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
    unsigned int numLayers = layeredMeshNumbering.numLayers;
    lcols.reserve(neq*this->numNodes*(numLayers+1));

    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

    // Loop over the sides that form the boundary condition
    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      lcols.resize(neq*numSideNodes*(numLayers+1));
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];

      GO base_id;
      for (unsigned int i = 0; i < numSideNodes; ++i) {
        std::size_t node = side.node[i];
        base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
        for (unsigned int il_col=0; il_col<numLayers+1; il_col++) {
          GO gnode = layeredMeshNumbering.getId(base_id, il_col);
          for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
            GO gcol  = solDOFManager.getGlobalDOF(gnode, eq_col);
            lcols[il_col*neq*numSideNodes + neq*i + eq_col] = solIndexer->getLocalElement(gcol);
          }
          if(il_col != fieldLevel) {
            const GO grow = solDOFManager.getGlobalDOF(gnode, this->offset); //insert diagonal values
            const LO lrow = solIndexer->getLocalElement(grow);
            Albany::setLocalRowValues(Jac,lrow,Teuchos::arrayView(&lrow,1), Teuchos::arrayView(&diagonal_value,1));
          }
        }
      }

      for (unsigned int i = 0; i < numSideNodes; ++i) {
        std::size_t node = side.node[i];
        base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](elem_LID,node) :
                    this->tensorRank == 1 ? this->valVec(elem_LID,node,eq) :
                    this->valTensor(elem_LID,node, eq/numDim, eq%numDim));
          // GO gnode = layeredMeshNumbering.getId(base_id, fieldLevel);
          // GO grow  = solDOFManager.getGlobalDOF(gnode,this->offset + eq);
          // const LO lrow = solIndexer->getLocalElement(grow);
          const LO lrow = nodeID(elem_LID,node,this->offset + eq);
          if (loadResid) {
            f_data[lrow] += valptr.val();
          }
          if (valptr.hasFastAccess()) {
            Albany::addToLocalRowValues(Jac,lrow,lcols(), Teuchos::arrayView(&(valptr.fastAccessDx(0)),lcols.size()));
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
  // Nothing to do here
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
  // Nothing to do here
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
ScatterResidual2D<PHAL::AlbanyTraits::HessianVec, Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{
  // Nothing to do here
}

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
  if (p.isType<int>("Offset 2D Field")) {
    offset2DField = p.get<int>("Offset 2D Field");
  } else {
    offset2DField = numFields-1;
  }
  fieldLevel = p.get<int>("Field Level");
}

// **********************************************************************
template<typename Traits>
void ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  const bool loadResid = Teuchos::nonnull(workset.f);
  const unsigned int neq = nodeID.extent(2);
  unsigned int nunk = this->numNodes*(neq-1);
  Teuchos::Array<LO> lcols, index;
  lcols.resize(nunk), index.resize(nunk);
  unsigned int numDim = 0;
  if (this->tensorRank==2) {
    numDim = this->valTensor.extent(2);
  }

  auto Jac = workset.Jac;
  Teuchos::ArrayRCP<ST> f_data;
  if (loadResid) {
    f_data = Albany::getNonconstLocalData(workset.f);
  }
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    // Local Unks: Loop over nodes in element, Loop over equations per node
    for (unsigned int node_col(0), i(0); node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        if(eq_col != offset2DField) {
          lcols[i] = nodeID(cell,node_col,eq_col);
          index[i++] = neq * node_col + eq_col;
        }
      }
    }
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if(this->offset + eq != offset2DField) {
          typename PHAL::Ref<ScalarT const>::type
            valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                      this->valTensor(cell,node, eq/numDim, eq%numDim));
          const LO lrow = nodeID(cell,node,this->offset + eq);
          if (loadResid) {
            f_data[lrow] += valptr.val();
          }
          // Check derivative array is nonzero
          if (valptr.hasFastAccess()) {
            // Sum Jacobian entries all at once
            for (unsigned int lunk = 0; lunk < nunk; lunk++) {
              Albany::addToLocalRowValues(Jac,lrow,
                                          Teuchos::arrayView(&lcols[lunk],1),
                                          Teuchos::arrayView(&(valptr.fastAccessDx(index[lunk])), 1));
            }
          } // has fast access
        }
      }
    }
  }

  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  auto solIndexer = workset.disc->getOverlapGlobalLocalIndexer();
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  lcols.resize(this->numNodes);

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  auto indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    Teuchos::ArrayRCP<LO> basalIds(this->numNodes);
    GO base_id;
    for (unsigned int node_col=0; node_col<this->numNodes; node_col++){
      base_id = layeredMeshNumbering.getColumnId(elNodeID[node_col]);
      GO gnode = layeredMeshNumbering.getId(base_id, fieldLevel);
      GO gcol  = solDOFManager.getGlobalDOF(gnode, offset2DField);
      lcols[node_col] = solIndexer->getLocalElement(gcol);
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if(eq != offset2DField) {
          const LO lrow = nodeID(cell,node,eq);
          typename PHAL::Ref<ScalarT const>::type valptr = this->valVec(cell,node,eq);

          if (loadResid) {
            f_data[lrow] += valptr.val();
          }
          if (valptr.hasFastAccess()) { // has fast access
              // Sum Jacobian entries all at once
            for (unsigned int i = 0; i<this->numNodes; ++i) {
              Albany::addToLocalRowValues(Jac,lrow,
                                              Teuchos::arrayView(&lcols[i],1),
                                              Teuchos::arrayView(&(valptr.fastAccessDx(neq*i + offset2DField)), 1));
            }
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
  // Nothing to do here
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
  // Nothing to do here
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
ScatterResidualWithExtrudedField<PHAL::AlbanyTraits::HessianVec, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>(p,dl)
{
  // Nothing to do here
}

} // namespace PHAL
