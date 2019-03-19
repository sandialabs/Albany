//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_NODAL_DATA_VECTOR_HPP
#define ADAPT_NODAL_DATA_VECTOR_HPP

#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_ThyraTypes.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Adapt_NodalFieldUtils.hpp"

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataVector
{
public:

  NodalDataVector(const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer,
                  NodeFieldSizeVector& nodeVectorLayout,
                  NodeFieldSizeMap& nodeVectorMap, LO& vectorsize);

  //! Destructor
  virtual ~NodalDataVector()  = default;

  // Methods to (re)build/replace vector spaces
  void replaceOwnedVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs);
  void replaceOwnedVectorSpace(const Teuchos::Array<GO>& owned_nodeGIDs,
                               const Teuchos::RCP<const Teuchos_Comm>& comm_);

  void replaceOverlapVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs);
  void replaceOverlapVectorSpace(const Teuchos::Array<GO>& overlap_nodeGIDs,
                                 const Teuchos::RCP<const Teuchos_Comm>& comm_);

  // Methods to get multivectors (or their data)
  const Teuchos::RCP<Thyra_MultiVector>& getOwnedNodeVector() const {
    return owned_node_vec;
  }
  const Teuchos::RCP<Thyra_MultiVector>& getOverlapNodeVector() const {
    return overlap_node_vec;
  }

  Teuchos::ArrayRCP<ST> getOwnedNodeView(std::size_t i) {
    return Albany::getNonconstLocalData(owned_node_vec->col(i));
  }
  Teuchos::ArrayRCP<ST> getOverlapNodeView(std::size_t i) {
    return Albany::getNonconstLocalData(overlap_node_vec->col(i));
  }

  Teuchos::ArrayRCP<const ST> getOwnedNodeConstView(std::size_t i) const {
    return Albany::getLocalData(owned_node_vec->col(i).getConst());
  }
  Teuchos::ArrayRCP<const ST> getOverlapNodeConstView(std::size_t i) const {
    return Albany::getLocalData(overlap_node_vec->col(i).getConst());
  }

  Teuchos::RCP<const Thyra_VectorSpace> getOverlappedVectorSpace() const { return overlap_node_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getOwnedVectorSpace()      const { return owned_node_vs;   }

  void initializeVectors (ST value);

  Teuchos::RCP<const Albany::CombineAndScatterManager> initializeCASManager ();

  void exportAddNodalDataVector();

  void saveNodalDataState() const;
  // In this version, mv may have fewer columns than there are vectors in the
  // database. start_col indicates the offset into the database.
  void saveNodalDataState(const Teuchos::RCP<const Thyra_MultiVector>& mv,
                          const int start_col) const;

  void saveNodalDataVector(const std::string& name,
                           const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec,
                           const int offset) const;

  void getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const;

  LO getVecSize() { return vectorsize; }

private:

  NodalDataVector();

  Teuchos::RCP<const Thyra_VectorSpace> overlap_node_vs;
  Teuchos::RCP<const Thyra_VectorSpace> owned_node_vs;

  Teuchos::RCP<Thyra_MultiVector> overlap_node_vec;
  Teuchos::RCP<Thyra_MultiVector> owned_node_vec;

  Teuchos::RCP<Albany::CombineAndScatterManager> cas_manager;

  Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

  NodeFieldSizeVector& nodeVectorLayout;
  NodeFieldSizeMap& nodeVectorMap;

  LO& vectorsize;

  bool mapsHaveChanged;

  int num_preeval_calls, num_posteval_calls;
};

} // namespace Adapt

#endif // ADAPT_NODAL_DATA_VECTOR_HPP
