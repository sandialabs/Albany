//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef __Albany_STKConnManager_hpp__
#define __Albany_STKConnManager_hpp__

#include <vector>

// Teuchos includes
#include "Teuchos_RCP.hpp"

// >Kokkos includes
#include "Kokkos_DynRankView.hpp"
#include "Kokkos_ViewFactory.hpp"

// Panzer includes
#include "Panzer_ConnManager.hpp"

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"

namespace Albany {

class STKConnManager : public panzer::ConnManager {
public:
   typedef typename panzer::ConnManager::LocalOrdinal LocalOrdinal;
   typedef typename panzer::ConnManager::GlobalOrdinal GlobalOrdinal;

   typedef double ProcIdData;
   typedef stk::mesh::Field<double> SolutionFieldType;
   typedef stk::mesh::Field<ProcIdData> ProcIdFieldType;
   typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType;

   // some simple exception classes
   struct ElementBlockException : public std::logic_error
   { ElementBlockException(const std::string & what) : std::logic_error(what) {} };

   struct SidesetException : public std::logic_error
   { SidesetException(const std::string & what) : std::logic_error(what) {} };

   STKConnManager(const Teuchos::RCP<AbstractSTKMeshStruct>& absMeshStruct);

   virtual ~STKConnManager() {}

   /** Tell the connection manager to build the connectivity assuming
     * a particular field pattern.
     *
     * \param[in] fp Field pattern to build connectivity for
     */
   virtual void buildConnectivity(const panzer::FieldPattern & fp);

   /** Build a clone of this connection manager, without any assumptions
     * about the required connectivity (e.g. <code>buildConnectivity</code>
     * has never been called).
     */
   virtual Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const;

   /** Add an element block with a string name
     */
//   void addElementBlock(const std::string & name,const CellTopologyData * ctData);

   /** Get ID connectivity for a particular element
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Pointer to beginning of indices, with total size
     *          equal to <code>getConnectivitySize(localElmtId)</code>
     */
   virtual const panzer::GlobalOrdinal * getConnectivity(LocalOrdinal localElmtId) const
   { return &connectivity_[elmtLidToConn_[localElmtId]]; }

   /** Get ID connectivity for a particular element
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Pointer to beginning of indices, with total size
     *          equal to <code>getConnectivitySize(localElmtId)</code>
     */
   virtual panzer::GlobalOrdinal * getConnectivity(LocalOrdinal localElmtId)
   { return &connectivity_[elmtLidToConn_[localElmtId]]; }

   /** How many mesh IDs are associated with this element?
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Number of mesh IDs that are associated with this element.
     */
   virtual LocalOrdinal getConnectivitySize(LocalOrdinal localElmtId) const
   { return connSize_[localElmtId]; }

   /** Get the block ID for a particular element.
     *
     * \param[in] localElmtId Local element ID
     */
   virtual std::string getBlockId(LocalOrdinal localElmtId) const;

   /** How many element blocks in this mesh?
     */
   virtual std::size_t numElementBlocks() const
   { return stkMeshStruct_->ebNames_.size(); }

   /** Get block IDs from STK mesh object
     */
   virtual void getElementBlockIds(std::vector<std::string> & elementBlockIds) const {

// Why is this different than stkDisc_->getElementBlockNames(blockIds);

        elementBlockIds.clear();
		for(std::size_t i = 0; i < stkMeshStruct_->ebNames_.size(); i++)

          elementBlockIds.push_back(stkMeshStruct_->ebNames_[i]);
   }

   virtual void getElementBlockNames(std::vector<std::string> & elementBlockIds) const {

// Why is this different than stkDisc_->getElementBlockNames(blockIds);

        elementBlockIds.clear();
		for(std::size_t i = 0; i < stkMeshStruct_->ebNames_.size(); i++)

          elementBlockIds.push_back(stkMeshStruct_->ebNames_[i]);
   }
      
   /** What are the cellTopologies linked to element blocks in this connection manager?
    */
   virtual void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const {

        elementBlockTopologies.clear();
		for(std::size_t i = 0; i < stkMeshStruct_->ebNames_.size(); i++)

          elementBlockTopologies.push_back(stkMeshStruct_->elementBlockTopologies_[i]);
   }
   /** Get the local element IDs for a paricular element
     * block. These are only the owned element ids.
     *
     * \param[in] blockIndex Block Index
     *
     * \returns Vector of local element IDs.
     */
   virtual const std::vector<LocalOrdinal> & getElementBlock(const std::string & blockId) const
   { return *(elementBlocks_.find(blockId)->second); }

   /** Get the local element IDs for a paricular element
     * block. These element ids are not owned, and the element
     * will live on another processor.
     *
     * \param[in] blockIndex Block Index
     *
     * \returns Vector of local element IDs.
     */
   virtual const std::vector<LocalOrdinal> & getNeighborElementBlock(const std::string & blockId) const
   { return *(neighborElementBlocks_.find(blockId)->second); }

   /** Get the coordinates (with local cell ids) for a specified element block and field pattern.
     *
     * \param[in] blockId Block containing the cells
     * \param[in] coordProvider Field pattern that builds the coordinates
     * \param[out] localCellIds Local cell Ids (indices)
     * \param[out] Resizable field container that contains the coordinates
     *             of the points on exit.
     */
   virtual void getDofCoords(const std::string & blockId,
                             const panzer::Intrepid2FieldPattern & coordProvider,
                             std::vector<std::size_t> & localCellIds,
                             Kokkos::DynRankView<double,PHX::Device> & points) const;

    /** How many elements are owned by this processor. Further,
      * the ordering of the local ids is suct that the first
      * <code>getOwnedElementCount()</code> elements are owned
      * by this processor. This is true only because of the
      * local element ids generated by the <code>STK_Interface</code>
      * object.
      */
    std::size_t getOwnedElementCount() const
    { return ownedElementCount_; }

    /** Before calling buildConnectivity, provide sideset IDs from which to
      * extract associated elements.
      */
    void associateElementsInSideset(const std::string sideset_id);

    /** After calling <code>buildConnectivity</code>, optionally check which
      * sidesets yielded no element associations in this communicator. This is a
      * parallel operation. In many applications, the outcome indicating
      * correctness is that the returned vector is empty.
      */
    std::vector<std::string> checkAssociateElementsInSidesets(const Teuchos::Comm<int>& comm) const;

    /** Get elements, if any, associated with <code>el</code>, excluding
      * <code>el</code> itself.
      */
    virtual const std::vector<LocalOrdinal>& getAssociatedNeighbors(const LocalOrdinal& el) const;

    /** Return whether getAssociatedNeighbors will return true for at least one
      * input. Default implementation returns false.
      */
    virtual bool hasAssociatedNeighbors() const;

   std::size_t elementLocalId(stk::mesh::Entity elmt) const;

   std::size_t elementLocalId(stk::mesh::EntityId gid) const;

   void getMyElements(std::vector<stk::mesh::Entity> & elements) const;

   void getMyElements(const std::string & blockID,std::vector<stk::mesh::Entity> & elements) const;

   /** Get a vector of elements that share an edge/face with an owned element. Note that these elements
     * are not owned.
     */
   void getNeighborElements(std::vector<stk::mesh::Entity> & elements) const;

   /** Get a vector of elements not owned by this processor but in a particular block
     */
   void getNeighborElements(const std::string & blockID,std::vector<stk::mesh::Entity> & elements) const;

   /**  Get the containing block ID of this element.
     */
   std::string containingBlockId(stk::mesh::Entity elmt) const;

//   std::pair<Teuchos::RCP<std::vector<std::pair<std::size_t,std::size_t> > >, Teuchos::RCP<std::vector<unsigned int> > >
//   getPeriodicNodePairing() const;

   /** Get set of element sharing a single node and its local node id.
     */
   void getOwnedElementsSharingNode(stk::mesh::Entity node,std::vector<stk::mesh::Entity> & elements,
                                    std::vector<int> & relIds) const;

   /** Get set of element sharing a single node and its local node id.
     */
//   void getOwnedElementsSharingNode(stk::mesh::EntityId nodeId,std::vector<stk::mesh::Entity> & elements,
//                                    std::vector<int> & relIds, unsigned int matchType) const;

   /** Get vertices and local cell IDs of a paricular element block.
     *
     * \param[in] mesh Reference to STK_Interface object
     * \param[in] blockId Element block identifier string
     * \param[out] localIds On processor local element IDs for the element block
     * \param[out] vertices Abstract array type (requires resize) containing
     *                      the coordinates of the vertices. Of size (#Cells, #Vertices, #Dim).
     */
   template<typename ArrayT>
   void getIdsAndVertices(
   		       std::string blockId,
   		       std::vector<std::size_t>& localIds,
   		       ArrayT& vertices) const;

   /** Get Entities corresponding to the locally owned part of the side set requested.
     * The Entities in the vector should be a dimension
     * lower then <code>getDimension()</code>.
     *
     * \param[in] sideName Name of side set
     * \param[in,out] sides Vector of entities containing the requested sides.
     */
   void getAllSides(const std::string & sideName,std::vector<stk::mesh::Entity> & sides) const;

   /** Get Entities corresponding to the side set requested. This also limits the entities
     * to be in a particular element block. The Entities in the vector should be a dimension
     * lower then <code>getDimension()</code>.
     *
     * \param[in] sideName Name of side set
     * \param[in] blockName Name of block
     * \param[in,out] sides Vector of entities containing the requested sides.
     */

   void getAllSides(const std::string & sideName,const std::string & blockName,std::vector<stk::mesh::Entity> & sides) const;

   //! get the block count
   stk::mesh::Part * getElementBlockPart(const std::string & name) const
   {
      std::map<std::string, stk::mesh::Part*>::const_iterator itr = stkMeshStruct_->elementBlockParts_.find(name);   // Element blocks
      if(itr==stkMeshStruct_->elementBlockParts_.end()) return 0;
      return itr->second;
   }

   stk::mesh::Part * getSideset(const std::string & name) const
   {
     auto itr = stkMeshStruct_->ssPartVec.find(name);
     return (itr != stkMeshStruct_->ssPartVec.end()) ? itr->second : nullptr;
   }

  /** Get the local element ID associated to the parent cell of a side element.
   * 
   * Functionality added to compute the graph associated to a block which has fields
   * defined in the volume and on the side.
   */
   std::size_t get_parent_cell_id(stk::mesh::Entity side) const;

   /** Get vertices associated with a number of elements of the same geometry.
     *
     * \param[in] localIds Element local IDs to construct vertices
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
//   template <typename ArrayT>
//   void getElementVertices(const std::vector<std::size_t> & localIds, ArrayT & vertices) const;

   /** Get vertices associated with a number of elements of the same geometry.
     *
     * \param[in] elements Element entities to construct vertices
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
//   template <typename ArrayT>
//   void getElementVertices(const std::vector<stk::mesh::Entity> & elements, ArrayT & vertices) const;

   /** Get vertices associated with a number of elements of the same geometry.
     *
     * \param[in] localIds Element local IDs to construct vertices
     * \param[in] eBlock Element block the elements are in
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
//   template <typename ArrayT>
//   void getElementVertices(const std::vector<std::size_t> & localIds,const std::string & eBlock, ArrayT & vertices) const;

   /** Get vertices associated with a number of elements of the same geometry.
     *
     * \param[in] elements Element entities to construct vertices
     * \param[in] eBlock Element block the elements are in
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
   template <typename ArrayT>
   void getElementVertices(const std::vector<stk::mesh::Entity> & elements,const std::string & eBlock, ArrayT & vertices) const;

   /** Get vertices associated with a number of elements of the same geometry. This access the true mesh coordinates
     * array.
     *
     * \param[in] elements Element entities to construct vertices
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
   template <typename ArrayT>
   void getElementVertices_FromCoords(const std::vector<stk::mesh::Entity> & elements, ArrayT & vertices) const;

   //! get the dimension
   unsigned getDimension() const
   { return dimension_; }

   /** Look up a global node and get the coordinate.
     */
   const double * getNodeCoordinates(stk::mesh::Entity node) const;

   /** Get vertices associated with a number of elements of the same geometry, note that a coordinate field
     * will be used (if not is available an exception will be thrown).
     *
     * \param[in] elements Element entities to construct vertices
     * \param[in] eBlock Element block the elements are in
     * \param[out] vertices Output array that will be sized (<code>localIds.size()</code>,#Vertices,#Dim)
     *
     * \note If not all elements have the same number of vertices an exception is thrown.
     *       If the size of <code>localIds</code> is 0, the function will silently return
     */
   template <typename ArrayT>
   void getElementVertices_FromField(const std::vector<stk::mesh::Entity> & elements,
          const std::string & eBlock, ArrayT & vertices) const;

   /** Get the stk mesh field pointer associated with a particular solution value
     * Assumes there is a field associated with "fieldName,blockId" pair. If none
     * is found an exception (std::runtime_error) is raised.
     */
   stk::mesh::Field<double> * getSolutionField(const std::string & fieldName,
                                               const std::string & blockId) const;

   stk::mesh::EntityRank getElementRank() const { return stk::topology::ELEMENT_RANK; }
   stk::mesh::EntityRank getSideRank() const { return metaData_->side_rank(); }
   stk::mesh::EntityRank getFaceRank() const { return stk::topology::FACE_RANK; }
   stk::mesh::EntityRank getEdgeRank() const { return stk::topology::EDGE_RANK; }
   stk::mesh::EntityRank getNodeRank() const { return stk::topology::NODE_RANK; }

   stk::mesh::EntityId getMaxEntityId(unsigned entityRank) const;

      //! get the global counts for the entity of specified rank
   std::size_t getEntityCounts(unsigned entityRank) const;

   /** Setup local element IDs
     */
   void buildLocalElementIDs();

  /** Get the EntityId associated to an element.
   * 
   * Functionality added to compute the graph associated to a block which has fields
   * defined in the volume and on the side.
   */
   stk::mesh::EntityId elementEntityId(stk::mesh::Entity elmt) const
   {
     return bulkData_->identifier(elmt);
   }

protected:
   /** Apply periodic boundary conditions associated with the mesh object.
     *
     * \note This function requires global All-2-All communication IFF
     *       periodic boundary conditions are required.
     */
//   void applyPeriodicBCs( const panzer::FieldPattern & fp, GlobalOrdinal nodeOffset, GlobalOrdinal edgeOffset,
//                                                           GlobalOrdinal faceOffset, GlobalOrdinal cellOffset);
   void applyInterfaceConditions();

   void buildLocalElementMapping();
   void clearLocalElementMapping();
   void buildOffsetsAndIdCounts(const panzer::FieldPattern & fp,
                                LocalOrdinal & nodeIdCnt, LocalOrdinal & edgeIdCnt,
                                LocalOrdinal & faceIdCnt, LocalOrdinal & cellIdCnt,
                                GlobalOrdinal & nodeOffset, GlobalOrdinal & edgeOffset,
                                GlobalOrdinal & faceOffset, GlobalOrdinal & cellOffset) const;

   LocalOrdinal addSubcellConnectivities(stk::mesh::Entity element,unsigned subcellRank,
                                         LocalOrdinal idCnt,GlobalOrdinal offset);

   void modifySubcellConnectivities(const panzer::FieldPattern & fp, stk::mesh::Entity element,
                                    unsigned subcellRank,unsigned subcellId,GlobalOrdinal newId,GlobalOrdinal offset);

   /** Compute global entity counts.
     */
   void buildEntityCounts();

   void buildMaxEntityIds();


   Teuchos::RCP<std::vector<stk::mesh::Entity> > elements_;

   // element block information
   std::map<std::string,Teuchos::RCP<std::vector<LocalOrdinal> > > elementBlocks_;
   std::map<std::string,Teuchos::RCP<std::vector<LocalOrdinal> > > neighborElementBlocks_;
   std::map<std::string,GlobalOrdinal> blockIdToIndex_;

   std::vector<LocalOrdinal> elmtLidToConn_; // element LID to Connectivity map
   std::vector<LocalOrdinal> connSize_; // element LID to Connectivity map
   std::vector<GlobalOrdinal> connectivity_; // Connectivity

   std::size_t ownedElementCount_;

   std::vector<std::string> sidesetsToAssociate_;
   std::vector<bool> sidesetYieldedAssociations_;
   std::vector<std::vector<LocalOrdinal> > elmtToAssociatedElmts_;

private:

   //! Stk Mesh Objects
   const Teuchos::RCP<stk::mesh::MetaData> metaData_;
   const Teuchos::RCP<stk::mesh::BulkData> bulkData_;

   const Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct_;

   int procRank_;
   std::size_t currentLocalId_;
   bool useFieldCoordinates_;
   unsigned dimension_;
   VectorFieldType * coordinatesField_;
   std::map<std::string,std::vector<std::string> > meshCoordFields_; // coordinate  fields written by user
   std::map<std::pair<std::string,std::string>,SolutionFieldType*> fieldNameToSolution_;

   // uses lazy evaluation
   mutable Teuchos::RCP<std::vector<stk::mesh::Entity> > orderedElementVector_;

   ProcIdFieldType * processorIdField_;

   // how many elements, faces, edges, and nodes are there globally
   std::vector<std::size_t> entityCounts_;

   // what is maximum entity ID
   std::vector<stk::mesh::EntityId> maxEntityId_;

   std::unordered_map<stk::mesh::EntityId, std::size_t> localIDHash_;

   /** Get Vector of element entities ordered by their LID, returns an RCP so that
     * it is easily stored by the caller.
     */
   Teuchos::RCP<const std::vector<stk::mesh::Entity> > getElementsOrderedByLID() const;


};

#if 0
template <typename ArrayT>
void STKConnManager::getElementVertices(const std::vector<std::size_t> & localElementIds, ArrayT & vertices) const
{
   if(!useFieldCoordinates_) {
     //
     // gather from the intrinsic mesh coordinates (non-lagrangian)
     //

     const std::vector<stk::mesh::Entity> & elements = *(this->getElementsOrderedByLID());

     // convert to a vector of entity objects
     std::vector<stk::mesh::Entity> selected_elements;
     for(std::size_t cell=0;cell<localElementIds.size();cell++)
       selected_elements.push_back(elements[localElementIds[cell]]);

     getElementVertices_FromCoords(selected_elements,vertices);
   }
   else {
     TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
                                "STK_Interface::getElementVertices: Cannot call this method when field coordinates are used "
                                "without specifying an element block.");
   }
}
#endif

template <typename ArrayT>
void STKConnManager::getElementVertices(const std::vector<stk::mesh::Entity> & elements,const std::string & eBlock, ArrayT & vertices) const
{
   if(!useFieldCoordinates_) {
     getElementVertices_FromCoords(elements,vertices);
   }
   else {
     getElementVertices_FromField(elements,eBlock,vertices);
   }
}

template <typename ArrayT>
void STKConnManager::getElementVertices_FromCoords(const std::vector<stk::mesh::Entity> & elements, ArrayT & vertices) const
{
   // nothing to do! silently return
   if(elements.size()==0) {
     vertices = Kokkos::createDynRankView(vertices,"vertices",0,0,0);
      return;
   }

   //
   // gather from the intrinsic mesh coordinates (non-lagrangian)
   //

   // get *master* cell toplogy...(belongs to first element)
   unsigned masterVertexCount
     = stk::mesh::get_cell_topology(bulkData_->bucket(elements[0]).topology()).getCellTopologyData()->vertex_count;

   // allocate space
   vertices = Kokkos::createDynRankView(vertices,"vertices",elements.size(),masterVertexCount,getDimension());

   // loop over each requested element
   unsigned dim = getDimension();
   for(std::size_t cell=0;cell<elements.size();cell++) {
      stk::mesh::Entity element = elements[cell];
      TEUCHOS_ASSERT(element!=0);

      unsigned vertexCount
        = stk::mesh::get_cell_topology(bulkData_->bucket(element).topology()).getCellTopologyData()->vertex_count;
      TEUCHOS_TEST_FOR_EXCEPTION(vertexCount!=masterVertexCount,std::runtime_error,
                         "In call to STK_Interface::getElementVertices all elements "
                         "must have the same vertex count!");

      // loop over all element nodes
      const size_t num_nodes = bulkData_->num_nodes(element);
      stk::mesh::Entity const* nodes = bulkData_->begin_nodes(element);
      TEUCHOS_TEST_FOR_EXCEPTION(num_nodes!=masterVertexCount,std::runtime_error,
                         "In call to STK_Interface::getElementVertices cardinality of "
                                 "element node relations must be the vertex count!");
      for(std::size_t node=0; node<num_nodes; ++node) {
        const double * coord = getNodeCoordinates(nodes[node]);

        // set each dimension of the coordinate
        for(unsigned d=0;d<dim;d++)
          vertices(cell,node,d) = coord[d];
      }
   }
}

template <typename ArrayT>
void STKConnManager::getElementVertices_FromField(const std::vector<stk::mesh::Entity> & elements,const std::string & eBlock, ArrayT & vertices) const
{
   TEUCHOS_ASSERT(useFieldCoordinates_);

   // nothing to do! silently return
   if(elements.size()==0) {
     vertices = Kokkos::createDynRankView(vertices,"vertices",0,0,0);
      return;
   }

   // get *master* cell toplogy...(belongs to first element)
   unsigned masterVertexCount
     = stk::mesh::get_cell_topology(bulkData_->bucket(elements[0]).topology()).getCellTopologyData()->vertex_count;

   // allocate space
   vertices = Kokkos::createDynRankView(vertices,"vertices",elements.size(),masterVertexCount,getDimension());

   std::map<std::string,std::vector<std::string> >::const_iterator itr = meshCoordFields_.find(eBlock);
   if(itr==meshCoordFields_.end()) {
     // no coordinate field set for this element block
     TEUCHOS_ASSERT(false);
   }

   const std::vector<std::string> & coordField = itr->second;
   std::vector<SolutionFieldType*> fields(getDimension());
   for(std::size_t d=0;d<fields.size();d++) {
     fields[d] = this->getSolutionField(coordField[d],eBlock);
   }

   for(std::size_t cell=0;cell<elements.size();cell++) {
      stk::mesh::Entity element = elements[cell];

      // loop over nodes set solution values
      const size_t num_nodes = bulkData_->num_nodes(element);
      stk::mesh::Entity const* nodes = bulkData_->begin_nodes(element);
      for(std::size_t i=0; i<num_nodes; ++i) {
        stk::mesh::Entity node = nodes[i];

        const double * coord = getNodeCoordinates(node);

        for(unsigned d=0;d<getDimension();d++) {
          double * solnData = stk::mesh::field_data(*fields[d],node);

          // recall mesh field coordinates are stored as displacements
          // from the mesh coordinates, make sure to add them together
          vertices(cell,i,d) = solnData[0]+coord[d];
        }
      }
   }
}


}

#endif
