////*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_SpatialFilter.hpp"
#include "ATO_Types.hpp"

#include "Albany_Application.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#ifdef ATO_USES_ISOLIB
#include "Albany_STKDiscretization.hpp"
#include "STKExtract.hpp"
#endif


namespace ATO
{

SpatialFilter::SpatialFilter (Teuchos::ParameterList& params)
 : m_filterRadius(params.get<double>("Filter Radius"))
{
  if( params.isType<Teuchos::Array<std::string> >("Blocks") ){
    m_blocks = params.get<Teuchos::Array<std::string> >("Blocks");
  }
  if( params.isType<int>("Iterations") ){
    m_iterations = params.get<int>("Iterations");
  } else {
    m_iterations = 1;
  }
}

void SpatialFilter::buildOperator (const app_type& app,
                                   const cas_type& cas_manager)
{
  const auto& wsElNodeID = app.getDiscretization()->getWsElNodeID();
  const auto& coords     = app.getDiscretization()->getCoords();
  const auto& wsEBNames  = app.getDiscretization()->getWsEBNames();

  // if this filter operates on a subset of the blocks in the mesh, create a list
  // of nodes that are not smoothed:
  std::set<int> excludeNodes;
  if (m_blocks.size() > 0){
    size_t num_worksets = coords.size();
    // add to the excludeNodes set all nodes that are not to be smoothed
    for (size_t ws=0; ws<num_worksets; ++ws) {
      if (std::find(m_blocks.begin(), m_blocks.end(), wsEBNames[ws]) != m_blocks.end()) {
        continue;
      }
      const int num_cells = coords[ws].size();
      for (int cell=0; cell<num_cells; ++cell) {
        const int num_nodes = coords[ws][cell].size();
        for (int node=0; node<num_nodes; node++) {
          const int gid = wsElNodeID[ws][cell][node];
          excludeNodes.insert(gid);
        }
      }
    }

    // remove from the excludeNodes set all nodes that are on boundaries 
    // between smoothed and non-smoothed blocks
    for (size_t ws=0; ws<num_worksets; ++ws) {
      if (std::find(m_blocks.begin(), m_blocks.end(), wsEBNames[ws]) == m_blocks.end()) {
        continue;
      }
      const int num_cells = coords[ws].size();
      for (int cell=0; cell<num_cells; cell++) {
        const int num_nodes = coords[ws][cell].size();
        for (int node=0; node<num_nodes; node++) {
          const int gid = wsElNodeID[ws][cell][node];
          const auto it = excludeNodes.find(gid);
          excludeNodes.erase(it,excludeNodes.end());
        }
      }
    }
  }

  nbrs_map_type neighbors;

  const double filter_radius_sqrd = m_filterRadius*m_filterRadius;

  // awful n^2 search... all against all
  const int dimension = app.getDiscretization()->getNumDim();
  const int num_worksets = coords.size();
  for (int home_ws=0; home_ws<num_worksets; ++home_ws) {
    const int home_num_cells = coords[home_ws].size();
    for (int home_cell=0; home_cell<home_num_cells; ++home_cell) {
      const int num_nodes = coords[home_ws][home_cell].size();
      for (int home_node=0; home_node<num_nodes; ++home_node) {
        GlobalPoint homeNode;
        homeNode.gid = wsElNodeID[home_ws][home_cell][home_node];
        if(neighbors.find(homeNode)==neighbors.end()) {
          // if this node was already accessed just skip
          for (int dim=0; dim<dimension; ++dim)  {
            homeNode.coords[dim] = coords[home_ws][home_cell][home_node][dim];
          }
          std::set<GlobalPoint> my_neighbors;
          if (excludeNodes.find(homeNode.gid) == excludeNodes.end()) {
            for (int trial_ws=0; trial_ws<num_worksets; ++trial_ws) {
              const auto it = std::find(m_blocks.begin(), m_blocks.end(), wsEBNames[trial_ws]);
              if (m_blocks.size() > 0 && it==m_blocks.end()) {
                continue;
              }
              const int trial_num_cells = coords[trial_ws].size();
              for (int trial_cell=0; trial_cell<trial_num_cells; ++trial_cell) {
                const int trial_num_nodes = coords[trial_ws][trial_cell].size();
                for (int trial_node=0; trial_node<trial_num_nodes; ++trial_node) {
                  const int gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                  if (excludeNodes.find(gid) != excludeNodes.end()) {
                    continue; // don't add excluded nodes
                  }
                  double tmp;
                  double delta_norm_sqr = 0.;
                  for (int dim=0; dim<dimension; ++dim)  {
                    //individual coordinates
                    tmp = homeNode.coords[dim]-coords[trial_ws][trial_cell][trial_node][dim];
                    delta_norm_sqr += tmp*tmp;
                  }
                  if (delta_norm_sqr<=filter_radius_sqrd) {
                    GlobalPoint newIntx;
                    newIntx.gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                    for (int dim=0; dim<dimension; ++dim) { 
                      newIntx.coords[dim] = coords[trial_ws][trial_cell][trial_node][dim];
                    }
                    my_neighbors.emplace(newIntx);
                  }
                }
              }
            }
          }
          neighbors.emplace(homeNode,my_neighbors);
        }
      }
    }
  }

  // communicate neighbor data
  importNeighbors(neighbors,cas_manager);
  
  // now build filter operator
  int numnonzeros = 0;
  const auto localNodeVS = cas_manager.getOwnedVectorSpace();

  // We first need to create the operator graph...
  Albany::ThyraCrsMatrixFactory opFactory(localNodeVS,localNodeVS,numnonzeros);
  auto spmd_localNodeVS = Albany::getSpmdVectorSpace(localNodeVS);
  for (const auto& it : neighbors) {
    const auto& homeNode = it.first;
    const GO home_node_gid = homeNode.gid;
    if (Albany::locallyOwnedComponent(spmd_localNodeVS,home_node_gid)) {
      const auto& connected_nodes = it.second;
      if (connected_nodes.size() > 0) {
        Teuchos::Array<GO> indices;
        indices.reserve(connected_nodes.size());
        for (const auto& connected_node : connected_nodes) {
          indices.push_back(connected_node.gid);
        }
        opFactory.insertGlobalIndices(home_node_gid,indices); 
      } else {
        // If the list of connected nodes is empty, still add a one on the diagonal.
        opFactory.insertGlobalIndices(home_node_gid,Teuchos::arrayView(&home_node_gid,1)); 
      }
    }
  }
  opFactory.fillComplete();

  // ... then we create the operator...
  m_filterOperator = opFactory.createOp();

  // ... and finally we fill it.
  Albany::resumeFill(m_filterOperator);
  for (const auto& it : neighbors) {
    const auto& homeNode = it.first;
    const GO home_node_gid = homeNode.gid;
    const auto& connected_nodes = it.second;
    Teuchos::Array<GO> indices;
    Teuchos::Array<ST> values;
    if (connected_nodes.size() > 0) {
      indices.reserve(connected_nodes.size());
      values.reserve(connected_nodes.size());
      for (const auto& connected_node : connected_nodes) {
        indices.push_back(connected_node.gid);
        const auto& nbr_coords = connected_node.coords;
        double distance = 0.0;
        for (int dim=0; dim<dimension; ++dim) {
          distance += (nbr_coords[dim]-homeNode.coords[dim])*(nbr_coords[dim]-homeNode.coords[dim]);
        }
        distance = (distance > 0.0) ? std::sqrt(distance) : 0.0;
        values.push_back(m_filterRadius - distance);
        Albany::addToGlobalRowValues(m_filterOperator,home_node_gid,indices(),values());
      }
    } else {
      // if the list of connected nodes is empty, still add a one on the diagonal.
      indices.push_back(home_node_gid);
      values.push_back(1.0);
      Albany::addToGlobalRowValues(m_filterOperator,home_node_gid,indices(),values());
    }
  }
  Albany::fillComplete(m_filterOperator);

  // Note: both Tpetra and Epetra Thyra adapters for LinearOp are derived from
  //       RowStatLinearOpBase and ScaledLinearOpBase, so the following is fine.
  auto rowSums = Thyra::createMember(m_filterOperator->range());
  auto stat_lop = Teuchos::rcp_dynamic_cast<Thyra::RowStatLinearOpBase<ST>>(m_filterOperator,true);
  stat_lop->getRowStat(Thyra::RowStatLinearOpBaseUtils::ROW_STAT_INV_ROW_SUM,rowSums.ptr());
  auto scaled_lop = Teuchos::rcp_dynamic_cast<Thyra::ScaledLinearOpBase<ST>>(m_filterOperator,true);
  scaled_lop->scaleLeft(*rowSums);
}

void SpatialFilter::importNeighbors(nbrs_map_type&  neighbors,
                                    const cas_type& cas_manager)
{
  // Aura points are the ones on the boundary (and may be owned or not by the current rank)
  const auto& owned_aura_vs  = cas_manager.getOwnedAuraVectorSpace();
  const auto& ghosted_aura_vs = cas_manager.getGhostedAuraVectorSpace();
  const auto ghosted_aura_gids = Albany::getGlobalElements(ghosted_aura_vs);
  const auto& ghosted_aura_owners = cas_manager.getGhostedAuraOwners();
  const auto& owned_aura_users = cas_manager.getOwnedAuraUsers();

  // Get from the cas manager the node global ids and the associated processor ids
  std::map<int, std::set<int> > boundaryNodesByProc;

  for (int i=0; i<ghosted_aura_owners.size(); ++i) {
    auto procIter = boundaryNodesByProc.find(ghosted_aura_owners[i]);
    GO gid = ghosted_aura_gids[i];
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(gid);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(ghosted_aura_owners[i],newSet) );
    } else {
      procIter->second.insert(gid);
    }
  }
  for (int i=0; i<owned_aura_users.size(); ++i) {
    auto procIter = boundaryNodesByProc.find(owned_aura_users[i].second);
    GO gid = Albany::getGlobalElement(owned_aura_vs,owned_aura_users[i].first);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(gid);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(owned_aura_users[i].second,newSet) );
    } else {
      procIter->second.insert(gid);
    }
  }


  int newPoints = 1;
  
  const MPI_Datatype MPI_GlobalPoint = get_MPI_GlobalPoint_type();
  while(newPoints > 0){
    newPoints = 0;

    int numNeighborProcs = boundaryNodesByProc.size();
    std::vector<std::vector<int> > numNeighbors_send(numNeighborProcs);
    std::vector<std::vector<int> > numNeighbors_recv(numNeighborProcs);
    
 
    // determine number of neighborhood nodes to be communicated
    int index = 0;
    std::map<int, std::set<int> >::iterator boundaryNodesIter;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      std::set<int>& boundaryNodes = boundaryNodesIter->second; 
      int numNodes = boundaryNodes.size();
  
      numNeighbors_send[index].resize(numNodes);
      numNeighbors_recv[index].resize(numNodes);
  
      GlobalPoint sendPoint;
      std::map< GlobalPoint, std::set<GlobalPoint> >::iterator sendPointIter;
      int localIndex = 0;
      std::set<int>::iterator boundaryNodeGID;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        std::set<GlobalPoint>& sendPointSet = sendPointIter->second;
        numNeighbors_send[index][localIndex] = sendPointSet.size();
        localIndex++;
      }
  
      MPI_Status status;
      MPI_Sendrecv(&(numNeighbors_send[index][0]), numNodes, MPI_INT, send_to, 0,
                   &(numNeighbors_recv[index][0]), numNodes, MPI_INT, recv_from, 0,
                   MPI_COMM_WORLD, &status);
      index++;
    }
  
    // new neighbors can't be immediately added to the neighbor map or they'll be
    // found and added to the list that's communicated to other procs.  This causes
    // problems because the message length has already been communicated.  
    std::map< GlobalPoint, std::set<GlobalPoint> > newNeighbors;
  
    // communicate neighborhood nodes
    index = 0;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      // determine total message size
      int totalNumEntries_send = 0;
      int totalNumEntries_recv = 0;
      std::vector<int>& send = numNeighbors_send[index];
      std::vector<int>& recv = numNeighbors_recv[index];
      int totalNumNodes = send.size();
      for(int i=0; i<totalNumNodes; i++){
        totalNumEntries_send += send[i];
        totalNumEntries_recv += recv[i];
      }
  
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      GlobalPoint* GlobalPoints_send = new GlobalPoint[totalNumEntries_send];
      GlobalPoint* GlobalPoints_recv = new GlobalPoint[totalNumEntries_recv];
      
      // copy into contiguous memory
      std::set<int>& boundaryNodes = boundaryNodesIter->second;
      GlobalPoint sendPoint;
      std::map< GlobalPoint, std::set<GlobalPoint> >::iterator sendPointIter;
      std::set<int>::iterator boundaryNodeGID;
      int offset = 0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        // get neighbors for boundary node i
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        TEUCHOS_TEST_FOR_EXCEPT( sendPointIter == neighbors.end() );
        std::set<GlobalPoint>& sendPointSet = sendPointIter->second;
        // copy neighbors into contiguous memory
        for(std::set<GlobalPoint>::iterator igp=sendPointSet.begin(); 
            igp!=sendPointSet.end(); igp++){
          GlobalPoints_send[offset] = *igp;
          offset++;
        }
      }
  
      MPI_Status status;
      MPI_Sendrecv(GlobalPoints_send, totalNumEntries_send, MPI_GlobalPoint, send_to, 0,
                   GlobalPoints_recv, totalNumEntries_recv, MPI_GlobalPoint, recv_from, 0,
                   MPI_COMM_WORLD, &status);
  
      // copy out of contiguous memory
      GlobalPoint recvPoint;
      std::map< GlobalPoint, std::set<GlobalPoint> >::iterator recvPointIter;
      offset = 0;
      int localIndex=0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        recvPoint.gid = *boundaryNodeGID;
        recvPointIter = newNeighbors.find(recvPoint);
        if( recvPointIter == newNeighbors.end() ){ // not found, add.
          std::set<GlobalPoint> newPointSet;
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            newPointSet.insert(GlobalPoints_recv[offset]);
            offset++;
          }
          newNeighbors.insert( std::pair<GlobalPoint,std::set<GlobalPoint> >(recvPoint,newPointSet) );
        } else {
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            recvPointIter->second.insert(GlobalPoints_recv[offset]);
            offset++;
          }
        }
        localIndex++;
      }
   
      delete [] GlobalPoints_send;
      delete [] GlobalPoints_recv;
      
      index++;
    }
  
    // add newNeighbors map to neighbors map
    std::map< GlobalPoint, std::set<GlobalPoint> >::iterator new_nbr;
    std::map< GlobalPoint, std::set<GlobalPoint> >::iterator nbr;
    std::set< GlobalPoint >::iterator newPoint;
    // loop on total neighbor list
    for(nbr=neighbors.begin(); nbr!=neighbors.end(); nbr++){
  
      std::set<GlobalPoint>& pointSet = nbr->second;
      int pointSetSize = pointSet.size();
  
      GlobalPoint home_point = nbr->first;
      double* home_coords = &(home_point.coords[0]);
      std::map< GlobalPoint, std::set<GlobalPoint> >::iterator nbrs;
      std::set< GlobalPoint >::iterator remote_point;
      for(nbrs=newNeighbors.begin(); nbrs!=newNeighbors.end(); nbrs++){
        std::set<GlobalPoint>& remote_points = nbrs->second;
        for(remote_point=remote_points.begin(); 
            remote_point!=remote_points.end();
            remote_point++){
          const double* remote_coords = &(remote_point->coords[0]);
          double distance = 0.0;
          for(int i=0; i<3; i++)
            distance += (remote_coords[i]-home_coords[i])*(remote_coords[i]-home_coords[i]);
          distance = (distance > 0.0) ? sqrt(distance) : 0.0;
          if( distance < m_filterRadius ) {
            pointSet.insert(*remote_point);
          }
        }
      }
      // see if any new points where found off processor.  
      newPoints += (pointSet.size() - pointSetSize);
    }
    int globalNewPoints=0;
    Teuchos::reduceAll(*(Albany::getComm(owned_aura_vs)), Teuchos::REDUCE_SUM, 1, &newPoints, &globalNewPoints); 
    newPoints = globalNewPoints;
  }
}

} // namespace ATO
