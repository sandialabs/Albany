//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "STKConnManager.hpp"

#include <vector>

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Comm.hpp>       // for comm_mesh_counts

#include "Teuchos_FancyOStream.hpp"

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;

// Object describing how to sort a vector of elements using
// local ID as the key
class LocalIdCompare {
public:

  LocalIdCompare(const STKConnManager& stk_disc) : stkDisc_(stk_disc) {}

  // Compares two stk mesh entities based on local ID
  bool operator() (stk::mesh::Entity a, stk::mesh::Entity b)
  { return stkDisc_.elementLocalId(a) < stkDisc_.elementLocalId(b);}

private:

  const STKConnManager&   stkDisc_;

};

STKConnManager::STKConnManager(const Teuchos::RCP<AbstractSTKMeshStruct>& absSTKMeshStruct)
   : ownedElementCount_(0), metaData_(absSTKMeshStruct->metaData),
     bulkData_(absSTKMeshStruct->bulkData), stkMeshStruct_(absSTKMeshStruct), useFieldCoordinates_(false)
{

     buildEntityCounts();
     buildMaxEntityIds();
     buildLocalElementIDs();

}

Teuchos::RCP<panzer::ConnManager>
STKConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new STKConnManager(stkMeshStruct_));
}

void STKConnManager::clearLocalElementMapping()
{
   elements_ = Teuchos::null;

   elementBlocks_.clear();
   elmtLidToConn_.clear();
   connSize_.clear();
   elmtToAssociatedElmts_.clear();
}

void STKConnManager::buildLocalElementMapping()
{
   clearLocalElementMapping(); // forget the past

   // build element block information
   //////////////////////////////////////////////
   elements_ = Teuchos::rcp(new std::vector<stk::mesh::Entity>);

   // defines ordering of blocks
   std::vector<std::string> blockIds;
//   stkDisc_->getElementBlockNames(blockIds);
   getElementBlockNames(blockIds);

   std::size_t blockIndex=0;
   for(std::vector<std::string>::const_iterator idItr=blockIds.begin();
       idItr!=blockIds.end();++idItr,++blockIndex) {
      std::string blockId = *idItr;

      // grab elements on this block
      std::vector<stk::mesh::Entity> blockElmts;
      getMyElements(blockId,blockElmts);

      // concatenate them into element LID lookup table
      elements_->insert(elements_->end(),blockElmts.begin(),blockElmts.end());

      // build block to LID map
      elementBlocks_[blockId] = Teuchos::rcp(new std::vector<LocalOrdinal>);
      for(std::size_t i=0;i<blockElmts.size();i++)
         elementBlocks_[blockId]->push_back(elementLocalId(blockElmts[i]));
   }

   ownedElementCount_ = elements_->size();

   blockIndex=0;
   for(std::vector<std::string>::const_iterator idItr=blockIds.begin();
       idItr!=blockIds.end();++idItr,++blockIndex) {
      std::string blockId = *idItr;

      // grab elements on this block
      std::vector<stk::mesh::Entity> blockElmts;
      getNeighborElements(blockId,blockElmts);

      // concatenate them into element LID lookup table
      elements_->insert(elements_->end(),blockElmts.begin(),blockElmts.end());

      // build block to LID map
      neighborElementBlocks_[blockId] = Teuchos::rcp(new std::vector<LocalOrdinal>);
      for(std::size_t i=0;i<blockElmts.size();i++)
         neighborElementBlocks_[blockId]->push_back(elementLocalId(blockElmts[i]));
   }

   // this expensive operation guarantees ordering of local IDs
   std::sort(elements_->begin(), elements_->end(), LocalIdCompare(*this));

   // allocate space for element LID to Connectivty map
   // connectivity size
   elmtLidToConn_.clear();
   elmtLidToConn_.resize(elements_->size(),0);

   connSize_.clear();
   connSize_.resize(elements_->size(),0);
}

void
STKConnManager::buildOffsetsAndIdCounts(const panzer::FieldPattern & fp,
                                        LocalOrdinal & nodeIdCnt, LocalOrdinal & edgeIdCnt,
                                        LocalOrdinal & faceIdCnt, LocalOrdinal & cellIdCnt,
                                        GlobalOrdinal & nodeOffset, GlobalOrdinal & edgeOffset,
                                        GlobalOrdinal & faceOffset, GlobalOrdinal & cellOffset) const
{
   // get the global counts for all the nodes, faces, edges and cells
   GlobalOrdinal maxNodeId = getMaxEntityId(getNodeRank());
   GlobalOrdinal maxEdgeId = getMaxEntityId(getEdgeRank());
   GlobalOrdinal maxFaceId = getMaxEntityId(getFaceRank());

   // compute ID counts for each sub cell type
   int patternDim = fp.getDimension();
   switch(patternDim) {
   case 3:
     faceIdCnt = fp.getSubcellIndices(2,0).size();
     // Intentional fall-through.
   case 2:
     edgeIdCnt = fp.getSubcellIndices(1,0).size();
     // Intentional fall-through.
   case 1:
     nodeIdCnt = fp.getSubcellIndices(0,0).size();
     cellIdCnt = fp.getSubcellIndices(patternDim,0).size();
     break;
   case 0:
   default:
      TEUCHOS_ASSERT(false);
   };

   // compute offsets for each sub cell type
   nodeOffset = 0;
   edgeOffset = nodeOffset+(maxNodeId+1)*nodeIdCnt;
   faceOffset = edgeOffset+(maxEdgeId+1)*edgeIdCnt;
   cellOffset = faceOffset+(maxFaceId+1)*faceIdCnt;

   // sanity check
   TEUCHOS_ASSERT(nodeOffset <= edgeOffset
               && edgeOffset <= faceOffset
               && faceOffset <= cellOffset);
}

STKConnManager::LocalOrdinal
STKConnManager::addSubcellConnectivities(stk::mesh::Entity element,
                                         unsigned subcellRank,
                                         LocalOrdinal idCnt,
                                         GlobalOrdinal offset)
{
   if(idCnt<=0)
      return 0 ;

   // loop over all relations of specified type
   LocalOrdinal numIds = 0;
   const stk::mesh::EntityRank rank = static_cast<stk::mesh::EntityRank>(subcellRank);
   const size_t num_rels = bulkData_->num_connectivity(element, rank);
   stk::mesh::Entity const* relations = bulkData_->begin(element, rank);
   for(std::size_t sc=0; sc<num_rels; ++sc) {
     stk::mesh::Entity subcell = relations[sc];

     // add connectivities: adjust for STK indexing craziness
     for(LocalOrdinal i=0;i<idCnt;i++)
       connectivity_.push_back(offset+idCnt*(bulkData_->identifier(subcell)-1)+i);

     numIds += idCnt;
   }
   return numIds;
}

void
STKConnManager::modifySubcellConnectivities(const panzer::FieldPattern & fp, stk::mesh::Entity element,
                                            unsigned subcellRank,unsigned subcellId,GlobalOrdinal newId,
                                            GlobalOrdinal offset)
{
   LocalOrdinal elmtLID = elementLocalId(element);
   auto * conn = this->getConnectivity(elmtLID);
   const std::vector<int> & subCellIndices = fp.getSubcellIndices(subcellRank,subcellId);

   // add connectivities: adjust for STK indexing craziness
   for(std::size_t i=0;i<subCellIndices.size();i++) {
      conn[subCellIndices[i]] = offset+subCellIndices.size()*(newId-1)+i;
   }
}

void STKConnManager::buildConnectivity(const panzer::FieldPattern & fp)
{
#ifdef HAVE_EXTRA_TIMERS
  using Teuchos::TimeMonitor;
  RCP<Teuchos::TimeMonitor> tM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(std::string("panzer_stk::STKConnManager::buildConnectivity"))));
#endif

   // get element info from STK_Interface
   // object and build a local element mapping.
   buildLocalElementMapping();

   // Build sub cell ID counts and offsets
   //    ID counts = How many IDs belong on each subcell (number of mesh DOF used)
   //    Offset = What is starting index for subcell ID type?
   //             Global numbering goes like [node ids, edge ids, face ids, cell ids]
   LocalOrdinal nodeIdCnt=0, edgeIdCnt=0, faceIdCnt=0, cellIdCnt=0;
   GlobalOrdinal nodeOffset=0, edgeOffset=0, faceOffset=0, cellOffset=0;
   buildOffsetsAndIdCounts(fp, nodeIdCnt,  edgeIdCnt,  faceIdCnt,  cellIdCnt,
                               nodeOffset, edgeOffset, faceOffset, cellOffset);

    // std::cout << "node: count = " << nodeIdCnt << ", offset = " << nodeOffset << std::endl;
    // std::cout << "edge: count = " << edgeIdCnt << ", offset = " << edgeOffset << std::endl;
    // std::cout << "face: count = " << faceIdCnt << ", offset = " << faceOffset << std::endl;
    // std::cout << "cell: count = " << cellIdCnt << ", offset = " << cellOffset << std::endl;

   // loop over elements and build global connectivity
   for(std::size_t elmtLid=0;elmtLid!=elements_->size();++elmtLid) {
      GlobalOrdinal numIds = 0;
      stk::mesh::Entity element = (*elements_)[elmtLid];

      // get index into connectivity array
      elmtLidToConn_[elmtLid] = connectivity_.size();

      // add connecviities for sub cells
      numIds += addSubcellConnectivities(element,getNodeRank(),nodeIdCnt,nodeOffset);
      numIds += addSubcellConnectivities(element,getEdgeRank(),edgeIdCnt,edgeOffset);
      numIds += addSubcellConnectivities(element,getFaceRank(),faceIdCnt,faceOffset);

      // add connectivity for parent cells
      if(cellIdCnt>0) {
         // add connectivities: adjust for STK indexing craziness
         for(LocalOrdinal i=0;i<cellIdCnt;i++)
            connectivity_.push_back(cellOffset+cellIdCnt*(bulkData_->identifier(element)-1));

         numIds += cellIdCnt;
      }

      connSize_[elmtLid] = numIds;
   }

//   applyPeriodicBCs( fp, nodeOffset, edgeOffset, faceOffset, cellOffset);

   // This method does not modify connectivity_. But it should be called here
   // because the data it initializes should be available at the same time as
   // connectivity_.
   if (hasAssociatedNeighbors())
     applyInterfaceConditions();
}

std::string STKConnManager::getBlockId(STKConnManager::LocalOrdinal localElmtId) const
{
   // walk through the element blocks and figure out which this ID belongs to
   stk::mesh::Entity element = (*elements_)[localElmtId];

   return containingBlockId(element);
}

#if 0
void STKConnManager::applyPeriodicBCs(const panzer::FieldPattern & fp, GlobalOrdinal nodeOffset, GlobalOrdinal edgeOffset,
                                      GlobalOrdinal faceOffset, GlobalOrdinal /* cellOffset */)
{
   using Teuchos::RCP;
   using Teuchos::rcp;

#ifdef HAVE_EXTRA_TIMERS
  using Teuchos::TimeMonitor;
  RCP<Teuchos::TimeMonitor> tM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(std::string("panzer_stk::STKConnManager::applyPeriodicBCs"))));
#endif

   std::pair<Teuchos::RCP<std::vector<std::pair<std::size_t,std::size_t> > >, Teuchos::RCP<std::vector<unsigned int> > > matchedValues
            = getPeriodicNodePairing();

   Teuchos::RCP<std::vector<std::pair<std::size_t,std::size_t> > > matchedNodes
            = matchedValues.first;
   Teuchos::RCP<std::vector<unsigned int> > matchTypes
            = matchedValues.second;

   // no matchedNodes means nothing to do!
   if(matchedNodes==Teuchos::null) return;

   for(std::size_t m=0;m<matchedNodes->size();m++) {
      stk::mesh::EntityId oldNodeId = (*matchedNodes)[m].first;
      std::size_t newNodeId = (*matchedNodes)[m].second;

      std::vector<stk::mesh::Entity> elements;
      std::vector<int> localIds;

      GlobalOrdinal offset0 = 0; // to make numbering consistent with that in PeriodicBC_Matcher
      GlobalOrdinal offset1 = 0; // offset for dof indexing
      if((*matchTypes)[m] == 0)
        offset1 = nodeOffset-offset0;
      else if((*matchTypes)[m] == 1){
        offset0 = getMaxEntityId(getNodeRank());
        offset1 = edgeOffset-offset0;
      } else if((*matchTypes)[m] == 2){
        offset0 = getMaxEntityId(getNodeRank())+getMaxEntityId(getEdgeRank());
        offset1 = faceOffset-offset0;
      } else
        TEUCHOS_ASSERT(false);

      // get relevent elements and node IDs
      getOwnedElementsSharingNode(oldNodeId-offset0,elements,localIds,(*matchTypes)[m]);

      // modify global numbering already built for each element
      for(std::size_t e=0;e<elements.size();e++){
         modifySubcellConnectivities(fp,elements[e],(*matchTypes)[m],localIds[e],newNodeId,offset1);
      }

   }
}
#endif

/** Get the coordinates for a specified element block and field pattern.
  */
void STKConnManager::getDofCoords(const std::string & blockId,
                                  const panzer::Intrepid2FieldPattern & coordProvider,
                                  std::vector<std::size_t> & localCellIds,
                                  Kokkos::DynRankView<double,PHX::Device> & points) const
{
   int dim = coordProvider.getDimension();
   int numIds = coordProvider.numberIds();

   // grab element vertices
   Kokkos::DynRankView<double,PHX::Device> vertices;
   getIdsAndVertices<Kokkos::DynRankView<double, PHX::Device> >(blockId, localCellIds, vertices);

   // setup output array
   points = Kokkos::DynRankView<double,PHX::Device>("points",localCellIds.size(),numIds,dim);
   coordProvider.getInterpolatoryCoordinates(vertices,points);
}

bool STKConnManager::hasAssociatedNeighbors() const
{
  return ! sidesetsToAssociate_.empty();
}

void STKConnManager::associateElementsInSideset(const std::string sideset_id)
{
  sidesetsToAssociate_.push_back(sideset_id);
  sidesetYieldedAssociations_.push_back(false);
}

inline std::size_t
getElementIdx(const std::vector<stk::mesh::Entity>& elements,
              stk::mesh::Entity const e)
{
  return static_cast<std::size_t>(
    std::distance(elements.begin(), std::find(elements.begin(), elements.end(), e)));
}

void STKConnManager::applyInterfaceConditions()
{
  elmtToAssociatedElmts_.resize(elements_->size());
  for (std::size_t i = 0; i < sidesetsToAssociate_.size(); ++i) {
    std::vector<stk::mesh::Entity> sides;
    getAllSides(sidesetsToAssociate_[i], sides);
    sidesetYieldedAssociations_[i] = ! sides.empty();
    for (std::vector<stk::mesh::Entity>::const_iterator si = sides.begin();
         si != sides.end(); ++si) {
      stk::mesh::Entity side = *si;
      const size_t num_elements = bulkData_->num_elements(side);
      stk::mesh::Entity const* elements = bulkData_->begin_elements(side);
      if (num_elements != 2) {
        // If relations.size() != 2 for one side in the sideset, then it's true
        // for all, including the first.
        TEUCHOS_ASSERT(si == sides.begin());
        sidesetYieldedAssociations_[i] = false;
        break;
      }
      const std::size_t ea_id = getElementIdx(*elements_, elements[0]),
        eb_id = getElementIdx(*elements_, elements[1]);
      elmtToAssociatedElmts_[ea_id].push_back(eb_id);
      elmtToAssociatedElmts_[eb_id].push_back(ea_id);
    }
  }
}

std::vector<std::string> STKConnManager::
checkAssociateElementsInSidesets(const Teuchos::Comm<int>& comm) const
{
  std::vector<std::string> sidesets;
  for (std::size_t i = 0; i < sidesetYieldedAssociations_.size(); ++i) {
    int sya, my_sya = sidesetYieldedAssociations_[i] ? 1 : 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &my_sya, &sya);
    if (sya == 0)
      sidesets.push_back(sidesetsToAssociate_[i]);
  }
  return sidesets;
}

const std::vector<STKConnManager::LocalOrdinal>&
STKConnManager::getAssociatedNeighbors(const LocalOrdinal& el) const
{
  return elmtToAssociatedElmts_[el];
}

void STKConnManager::buildEntityCounts()
{
   entityCounts_.clear();
   stk::mesh::comm_mesh_counts(*bulkData_,entityCounts_);
}

void STKConnManager::buildMaxEntityIds() {

      // developed to mirror "comm_mesh_counts" in stk_mesh/base/Comm.cpp
   
      const auto entityRankCount =  metaData_->entity_rank_count();
      const size_t   commCount        = 10; // entityRankCount
   
      TEUCHOS_ASSERT(entityRankCount<10);
   
      stk::ParallelMachine mach = bulkData_->parallel();
//      stk::ParallelMachine mach = stkDisc_*mpiComm_->getRawMpiComm();
      procRank_ = stk::parallel_machine_rank(mach);
   
      std::vector<stk::mesh::EntityId> local(commCount,0);
   
      // determine maximum ID for this processor for each entity type
      stk::mesh::Selector ownedPart = metaData_->locally_owned_part();
      for(stk::mesh::EntityRank i=stk::topology::NODE_RANK;
          i < static_cast<stk::mesh::EntityRank>(entityRankCount); ++i) {
         std::vector<stk::mesh::Entity> entities;
   
         stk::mesh::get_selected_entities(ownedPart, bulkData_->buckets(i), entities);
   
         // determine maximum ID for this processor
         std::vector<stk::mesh::Entity>::const_iterator itr;
         for(itr=entities.begin();itr!=entities.end();++itr) {
            stk::mesh::EntityId id = bulkData_->identifier(*itr);
            if(id>local[i])
               local[i] = id;
         }
      }
   
      // get largest IDs across processors
      stk::all_reduce(mach,stk::ReduceMax<10>(&local[0]));
      maxEntityId_.assign(local.begin(),local.begin()+entityRankCount+1);
}


std::size_t STKConnManager::elementLocalId(stk::mesh::Entity elmt) const
{
   return elementLocalId(bulkData_->identifier(elmt));
   // const std::size_t * fieldCoords = stk::mesh::field_data(*localIdField_,*elmt);
   // return fieldCoords[0];
}

std::size_t STKConnManager::elementLocalId(stk::mesh::EntityId gid) const
{
   // stk::mesh::EntityRank elementRank = getElementRank();
   // stk::mesh::Entity elmt = bulkData_->get_entity(elementRank,gid);
   // TEUCHOS_ASSERT(elmt->owner_rank()==procRank_);
   // return elementLocalId(elmt);
   std::unordered_map<stk::mesh::EntityId,std::size_t>::const_iterator itr = localIDHash_.find(gid);
   TEUCHOS_ASSERT(itr!=localIDHash_.end());
   return itr->second;
}

void STKConnManager::getMyElements(std::vector<stk::mesh::Entity> & elements) const
{
   // setup local ownership
   stk::mesh::Selector ownedPart = metaData_->locally_owned_part();

   // grab elements
   stk::mesh::EntityRank elementRank = getElementRank();
   stk::mesh::get_selected_entities(ownedPart, bulkData_->buckets(elementRank), elements);
}

void STKConnManager::getMyElements(const std::string & blockID,std::vector<stk::mesh::Entity> & elements) const
{
   stk::mesh::Part * elementBlock = getElementBlockPart(blockID);

   TEUCHOS_TEST_FOR_EXCEPTION(elementBlock==0,std::logic_error,"Could not find element block \"" << blockID << "\"");

   // setup local ownership
   // stk::mesh::Selector block = *elementBlock;
   stk::mesh::Selector ownedBlock = metaData_->locally_owned_part() & (*elementBlock);

   // grab elements
   stk::mesh::EntityRank elementRank = getElementRank();
   stk::mesh::get_selected_entities(ownedBlock, bulkData_->buckets(elementRank), elements);
}

void STKConnManager::getNeighborElements(std::vector<stk::mesh::Entity> & elements) const
{
   // setup local ownership
   stk::mesh::Selector neighborBlock = (!metaData_->locally_owned_part());

   // grab elements
   stk::mesh::EntityRank elementRank = getElementRank();
   stk::mesh::get_selected_entities(neighborBlock, bulkData_->buckets(elementRank), elements);
}

void STKConnManager::getNeighborElements(const std::string & blockID,std::vector<stk::mesh::Entity> & elements) const
{
   stk::mesh::Part * elementBlock = getElementBlockPart(blockID);

   TEUCHOS_TEST_FOR_EXCEPTION(elementBlock==0,std::logic_error,"Could not find element block \"" << blockID << "\"");

   // setup local ownership
   stk::mesh::Selector neighborBlock = (!metaData_->locally_owned_part()) & (*elementBlock);

   // grab elements
   stk::mesh::EntityRank elementRank = getElementRank();
   stk::mesh::get_selected_entities(neighborBlock, bulkData_->buckets(elementRank), elements);
}

std::size_t STKConnManager::getEntityCounts(unsigned entityRank) const
{
   TEUCHOS_TEST_FOR_EXCEPTION(entityRank>=entityCounts_.size(),std::logic_error,
                      "STKCOnnManager::getEntityCounts: Entity counts do not include rank: " << entityRank);

   return entityCounts_[entityRank];
}

stk::mesh::EntityId STKConnManager::getMaxEntityId(unsigned entityRank) const
{
   TEUCHOS_TEST_FOR_EXCEPTION(entityRank>=maxEntityId_.size(),std::logic_error,
                      "STK_Interface::getMaxEntityId: Max entity ids do not include rank: " << entityRank);

   return maxEntityId_[entityRank];
}

std::string STKConnManager::containingBlockId(stk::mesh::Entity elmt) const
{
   for(const auto & eb_pair : stkMeshStruct_->elementBlockParts_)
      if(bulkData_->bucket(elmt).member(*(eb_pair.second)))
         return eb_pair.first;
   return "";
}

#if 0
std::pair<Teuchos::RCP<std::vector<std::pair<std::size_t,std::size_t> > >, Teuchos::RCP<std::vector<unsigned int> > >
STKConnManager::getPeriodicNodePairing() const
{
   Teuchos::RCP<std::vector<std::pair<std::size_t,std::size_t> > > vec;
   Teuchos::RCP<std::vector<unsigned int > > type_vec = rcp(new std::vector<unsigned int>);
   const std::vector<Teuchos::RCP<const PeriodicBC_MatcherBase> > & matchers = getPeriodicBCVector();

   // build up the vectors by looping over the matched pair
   for(std::size_t m=0;m<matchers.size();m++){
      vec = matchers[m]->getMatchedPair(*this,vec);
      unsigned int type;
      if(matchers[m]->getType() == "coord")
        type = 0;
      else if(matchers[m]->getType() == "edge")
        type = 1;
      else if(matchers[m]->getType() == "face")
        type = 2;
      else
        TEUCHOS_ASSERT(false);
      type_vec->insert(type_vec->begin(),vec->size()-type_vec->size(),type);
   }

   return std::make_pair(vec,type_vec);

}
#endif

void STKConnManager::getOwnedElementsSharingNode(stk::mesh::Entity node,std::vector<stk::mesh::Entity> & elements,
                                                std::vector<int> & relIds) const
{
   // get all relations for node
   const size_t numElements = bulkData_->num_elements(node);
   stk::mesh::Entity const* relations = bulkData_->begin_elements(node);
   stk::mesh::ConnectivityOrdinal const* rel_ids = bulkData_->begin_element_ordinals(node);

   // extract elements sharing nodes
   for (size_t i = 0; i < numElements; ++i) {
      stk::mesh::Entity element = relations[i];

     // if owned by this processor
      if(bulkData_->parallel_owner_rank(element) == static_cast<int>(procRank_)) {
         elements.push_back(element);
         relIds.push_back(rel_ids[i]);
      }
   }
}

#if 0
void STKConnManager::getOwnedElementsSharingNode(stk::mesh::EntityId nodeId,std::vector<stk::mesh::Entity> & elements,
                                                                           std::vector<int> & relIds, unsigned int matchType) const
{
   stk::mesh::EntityRank rank;
   if(matchType == 0)
     rank = getNodeRank();
   else if(matchType == 1)
     rank = getEdgeRank();
   else if(matchType == 2)
     rank = getFaceRank();
   else
     TEUCHOS_ASSERT(false);

   stk::mesh::Entity node = bulkData_->get_entity(rank,nodeId);

   getOwnedElementsSharingNode(node,elements,relIds);
}
#endif

template<typename ArrayT>
void STKConnManager::getIdsAndVertices(
			 std::string blockId,
			 std::vector<std::size_t>& localIds,
			 ArrayT & vertices) const {
  
  std::vector<stk::mesh::Entity> elements;
  getMyElements(blockId,elements);
  
  // loop over elements of this block
  for(std::size_t elm=0;elm<elements.size();++elm) {
    stk::mesh::Entity element = elements[elm];
    
    localIds.push_back(elementLocalId(element));
  }

  // get vertices (this is slightly faster then the local id version)
  getElementVertices(elements,blockId,vertices);
}

void STKConnManager::getAllSides(const std::string & sideName,std::vector<stk::mesh::Entity> & sides) const
{
   stk::mesh::Part * sidePart = getSideset(sideName);
   TEUCHOS_TEST_FOR_EXCEPTION(sidePart==0,std::logic_error,
                      "Unknown side set \"" << sideName << "\"");

   stk::mesh::Selector side = *sidePart;

   // grab elements
   stk::mesh::get_selected_entities(side, bulkData_->buckets(getSideRank()), sides);
}

void STKConnManager::getAllSides(const std::string & sideName,const std::string & blockName,std::vector<stk::mesh::Entity> & sides) const
{
   stk::mesh::Part * sidePart = getSideset(sideName);
   stk::mesh::Part * elmtPart = getElementBlockPart(blockName);
   TEUCHOS_TEST_FOR_EXCEPTION(sidePart==0,SidesetException,
                      "Unknown side set \"" << sideName << "\"");
   TEUCHOS_TEST_FOR_EXCEPTION(elmtPart==0,ElementBlockException,
                      "Unknown element block \"" << blockName << "\"");

   stk::mesh::Selector side = *sidePart;
   stk::mesh::Selector block = *elmtPart;
   stk::mesh::Selector sideBlock = block & side;

   // grab elements
   stk::mesh::get_selected_entities(sideBlock, bulkData_->buckets(getSideRank()), sides);
}

Teuchos::RCP<const std::vector<stk::mesh::Entity> > 
STKConnManager::getElementsOrderedByLID() const
{
   using Teuchos::RCP;
   using Teuchos::rcp;

   if(orderedElementVector_==Teuchos::null) {
      // safe because essentially this is a call to modify a mutable object
      //const_cast<STK_Interface*>(this)->buildLocalElementIDs();
      const_cast<STKConnManager*>(this)->buildLocalElementIDs();
      //buildLocalElementIDs();
   }

   return orderedElementVector_.getConst();
}

void STKConnManager::buildLocalElementIDs()
{
   currentLocalId_ = 0;

   orderedElementVector_ = Teuchos::null; // forces rebuild of ordered lists

   // might be better (faster) to do this by buckets
   std::vector<stk::mesh::Entity> elements;
   getMyElements(elements);

   for(std::size_t index=0;index<elements.size();++index) {
      stk::mesh::Entity element = elements[index];

//GAH
      // set processor rank
//      ProcIdData * procId = stk::mesh::field_data(*processorIdField_,element);
//      procId[0] = Teuchos::as<ProcIdData>(procRank_);

      localIDHash_[bulkData_->identifier(element)] = currentLocalId_;

      currentLocalId_++;
   }

   // copy elements into the ordered element vector
   orderedElementVector_ = Teuchos::rcp(new std::vector<stk::mesh::Entity>(elements));

   elements.clear();
   getNeighborElements(elements);

   for(std::size_t index=0;index<elements.size();++index) {
      stk::mesh::Entity element = elements[index];

//GAH
      // set processor rank
//      ProcIdData * procId = stk::mesh::field_data(*processorIdField_,element);
//      procId[0] = Teuchos::as<ProcIdData>(procRank_);

      localIDHash_[bulkData_->identifier(element)] = currentLocalId_;

      currentLocalId_++;
   }

   orderedElementVector_->insert(orderedElementVector_->end(),elements.begin(),elements.end());
}

const double * STKConnManager::getNodeCoordinates(stk::mesh::Entity node) const
{
   return stk::mesh::field_data(*coordinatesField_,node);
}

stk::mesh::Field<double> * STKConnManager::getSolutionField(const std::string & fieldName,
                                                           const std::string & blockId) const
{
   // look up field in map
   std::map<std::pair<std::string,std::string>, SolutionFieldType*>::const_iterator
         iter = fieldNameToSolution_.find(std::make_pair(fieldName,blockId));

   // check to make sure field was actually found
   TEUCHOS_TEST_FOR_EXCEPTION(iter==fieldNameToSolution_.end(),std::runtime_error,
                      "Solution field name \"" << fieldName << "\" in block ID \"" << blockId << "\" was not found");

   return iter->second;
}

/* This is done in IOSSSTKMeshStruct now
void STKConnManager::addElementBlock(const std::string & name,const CellTopologyData * ctData)
{

   stk::mesh::Part * block = metaData_->get_part(name);
   if(block==0) {
     block = &metaData_->declare_part_with_topology(name, stk::mesh::get_topology(shards::CellTopology(ctData), dimension_));
   }

   // construct cell topology object for this block
   Teuchos::RCP<shards::CellTopology> ct
         = Teuchos::rcp(new shards::CellTopology(ctData));

   // add element block part and cell topology
   stkMeshStruct_->elementBlockParts_.insert(std::make_pair(name,block));
   stkMeshStruct_->elementBlockCT_.insert(std::make_pair(name,ct));
}
*/


}
