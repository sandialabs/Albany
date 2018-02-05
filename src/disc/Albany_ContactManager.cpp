//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_ContactManager.hpp"

#include "Moertel_InterfaceT.hpp"

const int printLevel = 4;

Albany::ContactManager::ContactManager(const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Albany::AbstractDiscretization& disc_,
	const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& meshSpecs_) :

	params(params_), disc(disc_), coordArray(disc_.getCoordinates()), meshSpecs(meshSpecs_)

{

  // Is contact specified?
  have_contact = params->isSublist("Contact");

  if(!have_contact) return;

  probDim = meshSpecs[0]->numDim;

  moertelManager = Teuchos::rcp( new MoertelT::ManagerT<ST, LO, Tpetra_GO, KokkosNode>(disc.getMapT()->getComm(), printLevel) );

  if(probDim == 2){
    oneD = true;
    moertelManager->SetDimension(MoertelT::ManagerT<ST, LO, Tpetra_GO, KokkosNode>::manager_2D);
  }
  else {
    oneD = false;
    moertelManager->SetDimension(MoertelT::ManagerT<ST, LO, Tpetra_GO, KokkosNode>::manager_3D);
  }

  moertelManager->SetProblemMap(disc.getMapT());

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

  if(disc.getMapT()->getComm()->getSize() != 2){ // Need exactly two ranks to write parallel rank output

    ALBANY_ASSERT(disc.getMapT()->getComm()->getSize() == 1, "ERROR - running in parallel but not using two ranks");
    // We are only serial here

    sfile.open ("slave_interface.txt");
    sfile << "This file contains the interface from sideset: " << slaveSideNames[0] << std::endl;
    mfile.open ("master_interface.txt");
    mfile << "This file contains the interface from sideset: " << masterSideNames[0] << std::endl;

  }
  else {
    int rank = disc.getMapT()->getComm()->getRank();
    std::ostringstream strstrm;
    strstrm << "slave_interface_" << rank << ".txt";
    sfile.open (strstrm.str());
    strstrm.str("");
    strstrm << "master_interface_" << rank << ".txt";
    mfile.open (strstrm.str());
  }

  const int mortarside(1);
  const int nonmortarside(0);

  const int number_of_mortar_pairs = masterSideNames.size();
  ALBANY_ASSERT(number_of_mortar_pairs == slaveSideNames.size(), "Input error: number of master and slave interfaces differ.");

  int interface_ctr = 0;

  // Loop over all the master interfaces
  for(int pair = 0; pair < number_of_mortar_pairs; pair++){

      processSS(interface_ctr, slaveSideNames[pair], 0 /* Slave side */, mortarside, sfile);

      interface_ctr++;

  }

  // Loop over all the slave interfaces
  for(int pair = 0; pair < number_of_mortar_pairs; pair++){

      processSS(interface_ctr, masterSideNames[pair], 1 /* mortar side */, nonmortarside, mfile);

      interface_ctr++;

  }

  // ============================================================= //
  // choose integration parameters
  // ============================================================= //

  Teuchos::ParameterList& moertelparams = moertelManager->Default_Parameters();

  // this does not affect this 2D case

  moertelparams.set("exact values at gauss points", true);

  // 1D interface possible values are 1,2,3,4,5,6,7,8,10 (2 recommended with linear shape functions)

  moertelparams.set("number gaussian points 1D", 2);

 // 2D interface possible values are 3,6,12,13,16,19,27

//    moertelparams.set("number gaussian points 2D",27);
  moertelparams.set("number gaussian points 2D", 12);  // 12 recommended with linear functions


  moertelManager->Integrate_Interfaces();

//   if (printlevel) cout << *manager;
  std::cout << *moertelManager;

  // GAH here

}

// Process all the contact surfaces and insert the data into a Moertel Interface
void
Albany::ContactManager::processSS(const int ctr, const std::string& sideSetName, int s_or_mortar, 
         int mortarside, std::ofstream& stream ){

  // one interface per side set name
  Teuchos::RCP<MoertelT::InterfaceT<ST, LO, Tpetra_GO, KokkosNode> > moertelInterface
     = Teuchos::rcp( new MoertelT::InterfaceT<ST, LO, Tpetra_GO, KokkosNode>(ctr, oneD, disc.getMapT()->getComm(), printLevel) );

  const MOERTEL::Function::FunctionType primal = MOERTEL::Function::func_Linear1D;
  const MOERTEL::Function::FunctionType dual = MOERTEL::Function::func_Linear1D/*func_Constant1D*/;

  moertelInterface->SetMortarSide(mortarside);
  moertelInterface->SetFunctionTypes(primal, dual);

//  std::size_t numFields = disc.getWsElNodeEqID()[0][0][0].size(); // num equations at each node
  std::size_t numFields = disc.getWsElNodeEqID()->dimension(3); // num equations at each node

  // Loop over all the worksets, and put the sides in the Moertel Interface Obj

  for(int workset = 0; workset < disc.getWsElNodeID().size(); workset++){

    const Albany::SideSetList& ssList = disc.getSideSets(workset);

    Albany::SideSetList::const_iterator it_side_set = ssList.find(sideSetName);

    // If ss exists in this workset, loop over the sides in it and construct moertel nodes/faces and interface
    if(it_side_set != ssList.end()){

      std::set<int> inserted_nodes;
      std::pair<std::set<int>::iterator, bool> ret;

      const std::vector<Albany::SideStruct>& theSideSet = it_side_set->second;

      for (std::size_t side=0; side < theSideSet.size(); ++side) {

        // Get the data that corresponds to the side.
        const int elem_GID   = theSideSet[side].elem_GID; // GID of the element that contains the master segment
        const int elem_LID   = theSideSet[side].elem_LID; // LID (numbered from zero) id of the element containing the
                                                     // master segment on this processor
        const int elem_side  = theSideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
        const int side_GID   = theSideSet[side].side_GID; // Need a global id for each contact "edge / face"
        const int elem_block = theSideSet[side].elem_ebIndex; // which  element block is the element in?
        const CellTopologyData_Subcell& subcell_side =  meshSpecs[elem_block]->ctd.side[elem_side];
        int numSideNodes = subcell_side.topology->node_count;

             stream << "side = " << side << std::endl;
             stream << "wsIndex = " << workset << std::endl;
             stream << "    element that owns side GID = " << elem_GID << std::endl;
             stream << "    element that owns side LID = " << elem_LID << std::endl;
             stream << "    side, local ID inside element = " << elem_side << std::endl;
             stream << "    side, global ID               = " << side_GID << std::endl;
             stream << "    element block side is in = " << elem_block << std::endl;

        // gather nodes from sideset and if unique then create moertel node
        const bool on_boundary = false; // will eventually want to allow boundaries to be intersected by contact surfaces
        const int  contact_pair_id = 0; // will eventually want to allow multiple pairs
        const Teuchos::ArrayRCP<GO>& elNodeID = disc.getWsElNodeID()[workset][elem_LID];
        const auto elNodeEqID = disc.getWsElNodeEqID()[workset];

        // loop over the nodes on the side

        std::vector<int> nodev(numSideNodes);

        for (int i = 0; i < numSideNodes; ++i) {
          std::size_t node = subcell_side.node[i];
          LO lnodeId = disc.getMapT()->getLocalElement(elNodeID[node]);
          GO gnodeId = disc.getMapT()->getGlobalElement(elNodeID[node]);
          nodev[i] = gnodeId;
          const double coords[] = { coordArray[3 * lnodeId],
               coordArray[3 * lnodeId + 1], 0.0 }; // Moertel node is 3 coords
//          const double coords[] = { coordArray[3 * lnodeId],
//               coordArray[3 * lnodeId + 1], coordArray[3 * lnodeid + 2] }; // Moertel node is 3 coords
          stream << "         node_LID = " << lnodeId << "   node_GID = " << gnodeId << "    node = " << node << std::endl;
          stream << "         coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << " - " << std::endl << std::endl;

  // Build the Moertel node list corresponding to unique nodes along the interface

          ret = inserted_nodes.insert(gnodeId);
//          const Teuchos::ArrayRCP<GO>& elNodeID = disc.getWsElNodeID()[workset][elem_LID];

          if (ret.second == true) { // this is a as yet unregistered node. add it

            std::vector<int> list_of_dofgid;

            for (std::size_t eq = 0; eq < numFields; eq++) {

              int global_eq_id = elNodeEqID(elem_LID, node, eq);
              list_of_dofgid.push_back(global_eq_id);

            }

            MOERTEL::Node moertel_node(gnodeId,
                                       coords, 
                                       list_of_dofgid.size(),
                                       &list_of_dofgid[0],
                                       on_boundary,
                                       printLevel);

            std::cout << "Adding node: " << gnodeId << "  to interface: " << ctr << std::endl;
            moertelInterface->AddNode(moertel_node, contact_pair_id);

          } // end add nodes to interface operations
        } // end loop over nodes on element side

  // 2D
        MOERTEL::Segment_Linear1D segment( side_GID, nodev, printLevel );
//  	  MOERTEL::Segment_BiLinearQuad segment( side_GID, nnodes, nodeid, printLevel ); // 3D
  	  moertelInterface->AddSegment(segment, s_or_mortar);

      }
    }
  }

  // once we have looped over all the worksets for this side set name, we close the interface
  ALBANY_ASSERT(moertelInterface->Complete() == true, "Contact interface is not complete");

  // add it to the manager
  moertelManager->AddInterface(moertelInterface);

}






 #if 0

  // Loop over the slave sides and construct moertel nodes/faces and interface
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
       for (std::size_t node=0; node < numNodes; ++node)
         for (std::size_t dim=0; dim < 3; ++dim)
             neumann(cell, node, dim) = 0.0; // zero out the accumulation vector

      const std::vector<Albany::SideStruct>& theSideSet = int_set_it->second;

      // Loop over the sides that form the boundary condition
      std::cout << "size of sideset array in workset = " << theSideSet.size() << std::endl;

         for (std::size_t side=0; side < theSideSet.size(); ++side) { // loop over the sides on this ws and name

           // Get the data that corresponds to the side.

           const int elem_GID = theSideSet[side].elem_GID; // GID of the element that contains the master segment
           const int elem_LID = theSideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
           const int elem_side = theSideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
           const int elem_block = theSideSet[side].elem_ebIndex; // which  element block is the element in?

           std::cout << "side = " << side << std::endl;
           std::cout << "    element that owns side GID = " << elem_GID << std::endl;
           std::cout << "    element that owns side LID = " << elem_LID << std::endl;
           std::cout << "    side, local ID inside element = " << elem_side << std::endl;
           std::cout << "    element block side is in = " << elem_block << std::endl << std::endl;


         }
#endif


  // Then assemble the DOFs (flux, traction) at the slaves into the master side local elements

#if 0  // Here is the assemble code, more or less

  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->val[eq])(cell,node);
    }
  } else
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->valVec[0])(cell,node,eq);
    }
  } else
  if (this->tensorRank == 2) {
    int numDims = this->valTensor[0].dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t i = 0; i < numDims; i++)
          for (std::size_t j = 0; j < numDims; j++)
            f_nonconstView[nodeID(cell,node,this->offset + i*numDims + j)] += (this->valTensor[0])(cell,node,i,j);

    }
  }
#endif

