//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_NODALDATABLOCK_HPP
#define ADAPT_NODALDATABLOCK_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Adapt_NodalFieldUtils.hpp"
#ifdef ALBANY_ATO
#include "Epetra_BlockMap.h"
#endif

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataBlock {

  public:

    NodalDataBlock(const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer,
                    NodeFieldSizeVector& nodeBlockLayout,
                    NodeFieldSizeMap& nodeBlockMap, LO& blocksize);

    //! Destructor
    virtual ~NodalDataBlock() {}

    void resizeLocalMap(const Teuchos::ArrayView<const GO>& local_nodeGIDs,
                        const Teuchos::RCP<const Teuchos_Comm>& comm_);

    void resizeOverlapMap(const Teuchos::ArrayView<const GO>& overlap_nodeGIDs,
                          const Teuchos::RCP<const Teuchos_Comm>& comm_);

    Teuchos::ArrayRCP<ST> getLocalNodeView() { return local_node_view; }
    Teuchos::ArrayRCP<ST> getOverlapNodeView() { return overlap_node_view; }

    Teuchos::ArrayRCP<const ST> getOverlapNodeConstView() const { return const_overlap_node_view; }
    Teuchos::ArrayRCP<const ST> getLocalNodeConstView() const { return const_local_node_view; }

    Teuchos::RCP<Tpetra_BlockMultiVector> getOverlapNodeVec() { return overlap_node_vec; }
    Teuchos::RCP<Tpetra_BlockMultiVector> getLocalNodeVec() { return local_node_vec; }

    Teuchos::RCP<const Tpetra_BlockMap> getOverlapMap() const { return overlap_node_map; }
    Teuchos::RCP<const Tpetra_BlockMap> getLocalMap() const { return local_node_map; }
#ifdef ALBANY_ATO
    Teuchos::RCP<const Epetra_BlockMap> getOverlapMapE() const { return overlap_node_mapE; }
    Teuchos::RCP<const Epetra_BlockMap> getLocalMapE() const { return local_node_mapE; }
#endif

    void initializeVectors(ST value) {
      overlap_node_vec->putScalar(value);
      local_node_vec->putScalar(value);
    }

    //eb-hack This interface, and the evaluator-based response functions that
    // interact with Exodus files through this and the Vector version of this
    // interface, need to be redesigned. There are number of problems. For
    // example, if there are multiple element blocks, multiple redundant calls
    // are made to these methods in preEvaluate and postEvaluate, possibly with
    // erroneous results.
    //   However, I want to continue to push off this task, so I'm expanding
    // eb-hack to take care of IPtoNodalField.
    void initEvaluateCalls();
    int numPreEvaluateCalls();
    int numPostEvaluateCalls();

    void initializeExport();

    void exportAddNodalDataBlock();

    void saveNodalDataState() const;

    void saveTpetraNodalDataVector(const std::string& name,
                                   const Teuchos::RCP<const Tpetra_Vector>& overlap_node_vec,
                                   int offset) const;

    LO getBlocksize(){ return blocksize; }
    
    void getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const;

    void registerState(const std::string &stateName, int ndofs);

    Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer(){ return nodeContainer; }


  private:

    NodalDataBlock(); 

    Teuchos::RCP<const Tpetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Tpetra_BlockMap> local_node_map;
#ifdef ALBANY_ATO
    Teuchos::RCP<const Epetra_BlockMap> overlap_node_mapE;
    Teuchos::RCP<const Epetra_BlockMap> local_node_mapE;
#endif

    Teuchos::RCP<Tpetra_BlockMultiVector> overlap_node_vec;
    Teuchos::RCP<Tpetra_BlockMultiVector> local_node_vec;

    Teuchos::RCP<Tpetra_Import> importer;

    Teuchos::ArrayRCP<ST> overlap_node_view;
    Teuchos::ArrayRCP<ST> local_node_view;
    Teuchos::ArrayRCP<const ST> const_overlap_node_view;
    Teuchos::ArrayRCP<const ST> const_local_node_view;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    NodeFieldSizeVector nodeBlockLayout;
    NodeFieldSizeMap nodeBlockMap;

    LO blocksize;

    bool mapsHaveChanged;

    int num_preeval_calls, num_posteval_calls;

};

}

#endif // ADAPT_NODALDATABLOCK_HPP
