//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ContactManager.hpp"



Albany::ContactManager::ContactManager(const Teuchos::RCP<Teuchos::ParameterList>& params_) :
	params(params_)
{

  // Is contact specified?
  have_contact = params->isSublist("Contact");

  if(!have_contact) return;

  Teuchos::ParameterList& paramList = params->sublist("Contact");

  masterSideNames =
        paramList.get<Teuchos::Array<std::string>>("Master Side Sets");
  slaveSideNames =
        paramList.get<Teuchos::Array<std::string>>("Slave Side Sets");
  sideSetIDs =
        paramList.get<Teuchos::Array<std::string>>("Contact Side Set Pair");
  constrainedFields =
        paramList.get<Teuchos::Array<std::string>>("Constrained Field Names");

  // Print names of field variables to be constrainted
  std::cout << "Number of constrained fields: " << constrainedFields.size() << std::endl;
  for(std::size_t i = 0; i < constrainedFields.size(); i++)
    std::cout << constrainedFields[i] << std::endl;

  // Print the master side set names
  std::cout << "Master side sets found, number = " << masterSideNames.size() << std::endl;
  for(int i = 0; i < masterSideNames.size(); i++)
    std::cout << masterSideNames[i] << std::endl;

  // Print the slave side set names
  std::cout << "Slave side sets found, number = " << slaveSideNames.size() << std::endl;
  for(int i = 0; i < slaveSideNames.size(); i++)
    std::cout << slaveSideNames[i] << std::endl;

  // Print all sideset ids
  size_t num_contact_pairs = sideSetIDs.size()/2;
  std::cout << "Number of contact pairs as master/slave:" << num_contact_pairs << std::endl;
  for(size_t i = 0; i < num_contact_pairs; i+=2)
    std::cout << sideSetIDs[i] << "/" << sideSetIDs[i+1] << std::endl;

  sfile.open ("slave_interface.txt");
  mfile.open ("master_interface.txt");


}

void
Albany::ContactManager::initializeContactSurfaces(const std::vector<Albany::SideSetList>& ssListVec,
							const Teuchos::ArrayRCP<double>& coordArray,
							const Teuchos::RCP<const Tpetra_Map>& overlap_node_mapT,
							const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& wsElNodeID,
							const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& meshSpecs){

  if(!have_contact) return; // do nothing if contact has not been specified in the problem

  // Loop over all the worksets, and put the sides in the Moertel Interface Obj

  for(int workset = 0; workset < ssListVec.size(); workset++){

    const Albany::SideSetList& ssList = ssListVec[workset];

    Albany::SideSetList::const_iterator it_master = ssList.find(masterSideNames[0]);
    Albany::SideSetList::const_iterator it_slave  = ssList.find(slaveSideNames[0]);

    const std::vector<Albany::SideStruct>& slaveSideSet = it_slave->second;
    const std::vector<Albany::SideStruct>& masterSideSet = it_master->second;

    // num overlapped nodes
    int num_nodes = coordArray.size() / 3;
    std::size_t numFields = 2; //hack

  // If slave ss exists, loop over the slave sides and construct moertel nodes/faces and interface
  if(it_slave != ssList.end()) {
    std::set<int> inserted_nodes;
    std::set<int>::iterator it;
    std::pair<std::set<int>::iterator, bool> ret;
    for (std::size_t side=0; side < slaveSideSet.size(); ++side) {

      // Get the data that corresponds to the side. 
      const int elem_GID   = slaveSideSet[side].elem_GID; // GID of the element that contains the master segment
      const int elem_LID   = slaveSideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
      const int elem_side  = slaveSideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
      const int elem_block = slaveSideSet[side].elem_ebIndex; // which  element block is the element in?
      const CellTopologyData_Subcell& subcell_side =  meshSpecs[elem_block]->ctd.side[elem_side];
      int numSideNodes = subcell_side.topology->node_count;

           sfile << "side = " << side << std::endl;
           mfile << "wsIndex = " << workset << std::endl;
           sfile << "    element that owns side GID = " << elem_GID << std::endl;
           sfile << "    element that owns side LID = " << elem_LID << std::endl;
           sfile << "    side, local ID inside element = " << elem_side << std::endl;
           sfile << "    element block side is in = " << elem_block << std::endl;

      // gather nodes from sideset and if unique then create moertel node
      const int  print_level = 4;     // experience from ALEGRA suggests this is a good choice... 
                                      // ... probably will want to parse this in production code
      const bool on_boundary = false; // will eventually want to allow boundaries to be intersected by contact surfaces
      const int  contact_pair_id = 0; // will eventually want to allow multiple pairs
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[workset][elem_LID];
      for (int i = 0; i < numSideNodes; ++i) {
        std::size_t node = subcell_side.node[i];
        LO lnodeId = overlap_node_mapT->getLocalElement(elNodeID[node]);
        GO gnodeId = overlap_node_mapT->getGlobalElement(elNodeID[node]);
        const double coords[] = { coordArray[3 * lnodeId], 
             coordArray[3 * lnodeId + 1], 0.0 }; // Moertel node is 3 coords
//        const double coords[] = { coordArray[3 * lnodeId], 
//             coordArray[3 * lnodeId + 1], coordArray[3 * lnodeid + 2] }; // Moertel node is 3 coords
        sfile << "         node_LID = " << lnodeId << "   node_GID = " << gnodeId << "    node = " << node << std::endl;
        sfile << "         coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << " - " << std::endl << std::endl;
      }
/*
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> wsElNodeID = workset.wsElNodeID;
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> &wsElNodeEqID = workset.wsElNodeEqID[elem_LID];
      for (std::size_t node=0; node < num_nodes; ++node) {
        ret = inserted_nodes.insert(wsElNodeID[elem_LID][node]);
        if (ret.second==true) { // this is a as yet unregistered node. add it
          const double coords[] = { this->coordVec(elem_LID, node, 0), 
                this->coordVec(elem_LID, node, 1), 0.0 }; // Moertel node is 3 coords
          std::vector<int> list_of_dofgid;
          for (std::size_t eq=0; eq < numFields; eq++) {
            int global_eq_id = wsElNodeEqID[node][eq];
            list_of_dofgid.push_back(global_eq_id);
          }
*/

/*
          MOERTEL::Node moertel_node(wsElNodeID[elem_LID][node], 
                                     coords, list_of_dofgid.size(), 
                                     &list_of_dofgid[0], 
                                     on_boundary, 
                                     print_level);
          _moertelInterface->AddNode(moertel_node,contact_pair_id);
        }
      }
*/
    }
  }
  
  if(it_master != ssList.end()) {
      //repeat slave side setup here.
    std::set<int> inserted_nodes;
    std::set<int>::iterator it;
    std::pair<std::set<int>::iterator, bool> ret;
    for (std::size_t side=0; side < masterSideSet.size(); ++side) {

      // Get the data that corresponds to the side. 
      const int elem_GID   = masterSideSet[side].elem_GID; // GID of the element that contains the master segment
      const int elem_LID   = masterSideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
      const int elem_side  = masterSideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
      const int elem_block = masterSideSet[side].elem_ebIndex; // which  element block is the element in?
      const CellTopologyData_Subcell& subcell_side =  meshSpecs[elem_block]->ctd.side[elem_side];
      int numSideNodes = subcell_side.topology->node_count;

           mfile << "side = " << side << std::endl;
           mfile << "wsIndex = " << workset << std::endl;
           mfile << "    element that owns side GID = " << elem_GID << std::endl;
           mfile << "    element that owns side LID = " << elem_LID << std::endl;
           mfile << "    side, local ID inside element = " << elem_side << std::endl;
           mfile << "    element block side is in = " << elem_block << std::endl;

      // gather nodes from sideset and if unique then create moertel node
      const int  print_level = 4;     // experience from ALEGRA suggests this is a good choice... 
                                      // ... probably will want to parse this in production code
      const bool on_boundary = false; // will eventually want to allow boundaries to be intersected by contact surfaces
      const int  contact_pair_id = 0; // will eventually want to allow multiple pairs
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[workset][elem_LID];
      for (int i = 0; i < numSideNodes; ++i) {
        std::size_t node = subcell_side.node[i];
        LO lnodeId = overlap_node_mapT->getLocalElement(elNodeID[node]);
        GO gnodeId = overlap_node_mapT->getGlobalElement(elNodeID[node]);
        const double coords[] = { coordArray[3 * lnodeId], 
             coordArray[3 * lnodeId + 1], 0.0 }; // Moertel node is 3 coords
        mfile << "         node_LID = " << lnodeId << "   node_GID = " << gnodeId << "    node = " << node << std::endl;
        mfile << "         coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << " - " << std::endl << std::endl;
      }
    }
  }


//  this->pContactObj->fillMandD(workset);

 #if 0

  // Loop over the slave sides and construct moertel nodes/faces and interface
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
       for (std::size_t node=0; node < numNodes; ++node)
         for (std::size_t dim=0; dim < 3; ++dim)
             neumann(cell, node, dim) = 0.0; // zero out the accumulation vector

      const std::vector<Albany::SideStruct>& sideSet = it->second;

      // Loop over the sides that form the boundary condition
      std::cout << "size of sideset array in workset = " << sideSet.size() << std::endl;
      
         for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

           // Get the data that corresponds to the side. 
         
           const int elem_GID = sideSet[side].elem_GID; // GID of the element that contains the master segment
           const int elem_LID = sideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
           const int elem_side = sideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
           const int elem_block = sideSet[side].elem_ebIndex; // which  element block is the element in?
 
           std::cout << "side = " << side << std::endl;
           std::cout << "    element that owns side GID = " << elem_GID << std::endl;
           std::cout << "    element that owns side LID = " << elem_LID << std::endl;
           std::cout << "    side, local ID inside element = " << elem_side << std::endl;
           std::cout << "    element block side is in = " << elem_block << std::endl << std::endl;


         }
#endif


  // Then assemble the DOFs (flux, traction) at the slaves into the master side local elements

#if 0  // Here is the assemble code, more or less

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->val[eq])(cell,node);
    }
  } else 
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->valVec[0])(cell,node,eq);
    }
  } else
  if (this->tensorRank == 2) {
    int numDims = this->valTensor[0].dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t i = 0; i < numDims; i++)
          for (std::size_t j = 0; j < numDims; j++)
            f_nonconstView[nodeID[node][this->offset + i*numDims + j]] += (this->valTensor[0])(cell,node,i,j);
  
    }
  }
#endif
}

}

