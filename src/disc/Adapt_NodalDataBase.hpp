//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


//IK, 9/12/14: no Epetra!
#ifndef ADAPT_NODALDATABASE_HPP
#define ADAPT_NODALDATABASE_HPP


#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Adapt_NodalFieldUtils.hpp"

namespace Adapt {

class NodalDataVector;

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataBase {

  public:

    NodalDataBase();

    //! Destructor
    virtual ~NodalDataBase(){}

    Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer(){ return nodeContainer; }

    void updateNodalGraph(const Teuchos::RCP<const Tpetra_CrsGraph>& nGraph)
         { nodalGraph = nGraph; }

    const Teuchos::RCP<const Tpetra_CrsGraph>& getNodalGraph()
         { return nodalGraph; }

    void resizeLocalMap(const Teuchos::Array<GO>& local_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    void resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    bool isNodeDataPresent() { return Teuchos::nonnull(nodal_data_vector); }

    void registerVectorState(const std::string &stateName, int ndofs);

    LO getVecsize(){ return vectorsize; }

    Teuchos::RCP<Adapt::NodalDataVector> getNodalDataVector() {
      TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(nodal_data_vector), std::logic_error,
         "Nodal Data Base: Error - nodal_data_vector has not been allocated!" << std::endl);
      return nodal_data_vector;
    }

  private:

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    Teuchos::RCP<const Tpetra_CrsGraph> nodalGraph;

    NodeFieldSizeVector nodeVectorLayout;
    NodeFieldSizeMap nodeVectorMap;

    LO vectorsize;

    Teuchos::RCP<Adapt::NodalDataVector> nodal_data_vector;

    void initialize();

    bool initialized;

};


}

#endif // ADAPT_NODALDATABASE_HPP
