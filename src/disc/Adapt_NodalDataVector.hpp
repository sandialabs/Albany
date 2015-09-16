//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ADAPT_NODALDATAVECTOR_HPP
#define ADAPT_NODALDATAVECTOR_HPP

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
class NodalDataVector {

  public:

    NodalDataVector(const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer,
                    NodeFieldSizeVector& nodeVectorLayout,
                    NodeFieldSizeMap& nodeVectorMap, LO& vectorsize);

    //! Destructor
    virtual ~NodalDataVector() {}

    void resizeLocalMap(const Teuchos::Array<GO>& local_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    void resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    const Teuchos::RCP<Tpetra_MultiVector>& getLocalNodeVector() const {
      return local_node_vec;
    }
    const Teuchos::RCP<Tpetra_MultiVector>& getOverlapNodeVector() const {
      return overlap_node_vec;
    }

    Teuchos::ArrayRCP<ST> getLocalNodeView(std::size_t i) {
      return local_node_vec->getVectorNonConst(i)->get1dViewNonConst();
    }
    Teuchos::ArrayRCP<ST> getOverlapNodeView(std::size_t i) {
      return overlap_node_vec->getVectorNonConst(i)->get1dViewNonConst();
    }

    Teuchos::ArrayRCP<const ST> getOverlapNodeConstView(std::size_t i) const {
      return overlap_node_vec->getVector(i)->get1dView();
    }
    Teuchos::ArrayRCP<const ST> getLocalNodeConstView(std::size_t i) const {
      return local_node_vec->getVector(i)->get1dView();
    }

    Teuchos::RCP<const Tpetra_Map> getOverlapMap() const { return overlap_node_map; }
    Teuchos::RCP<const Tpetra_Map> getLocalMap() const { return local_node_map; }

#ifdef ALBANY_ATO
    Teuchos::RCP<const Epetra_BlockMap> getOverlapBlockMapE() const { return overlap_node_mapE; }
    Teuchos::RCP<const Epetra_BlockMap> getLocalBlockMapE() const { return local_node_mapE; }
#endif

    void initializeVectors(ST value);

    void initializeExport();

    void exportAddNodalDataVector();

    void saveNodalDataState() const;
    // In this version, mv may have fewer columns than there are vectors in the
    // database. start_col indicates the offset into the database.
    void saveNodalDataState(const Teuchos::RCP<const Tpetra_MultiVector>& mv,
                            const int start_col) const;

  void saveTpetraNodalDataVector(
    const std::string& name,
    const Teuchos::RCP<const Tpetra_MultiVector>& overlap_node_vec,
    const int offset) const;

    void getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const;

    LO getVecSize() { return vectorsize; }

  private:

    NodalDataVector();

    Teuchos::RCP<const Tpetra_Map> overlap_node_map;
    Teuchos::RCP<const Tpetra_Map> local_node_map;

    Teuchos::RCP<Tpetra_MultiVector> overlap_node_vec;
    Teuchos::RCP<Tpetra_MultiVector> local_node_vec;

#ifdef ALBANY_ATO
    Teuchos::RCP<const Epetra_BlockMap> overlap_node_mapE;
    Teuchos::RCP<const Epetra_BlockMap> local_node_mapE;
#endif

    Teuchos::RCP<Tpetra_Import> importer;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    NodeFieldSizeVector& nodeVectorLayout;
    NodeFieldSizeMap& nodeVectorMap;

    LO& vectorsize;

    bool mapsHaveChanged;

    int num_preeval_calls, num_posteval_calls;

};

}

#endif // ADAPT_NODALDATABLOCK_HPP
