//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Utils.hpp"

#include "mrtr_interface.H"
#include "mrtr_pnode.H"

#include <set>

// **********************************************************************
// Constructor
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
MortarContact<EvalT, Traits>::
MortarContact(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :

  meshSpecs         (p.get<const Albany::MeshSpecsStruct*>("Mesh Specs Struct")),
  masterSideNames   (p.get<Teuchos::Array<std::string>>("Master Side Set Names")), 
  slaveSideNames    (p.get<Teuchos::Array<std::string>>("Slave Side Set Names")), 
  sideSetIDs        (p.get<Teuchos::Array<std::string>>("Sideset IDs")), //array of sidesets
  coordVec          (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector), //Node coords
  M_operator        (p.get<std::string>("M Name"), dl->qp_scalar),  //M portion of G
  constrainedFields (p.get<Teuchos::Array<std::string>>("Constrained Field Names")) //Names of fields to be constrained

{

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

  // This evaluator uses the nodal coordinates to form the M and D operator
  this->addDependentField(coordVec);
  this->addEvaluatedField(M_operator);

  this->setName("Mortar Contact Constraints"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void MortarContact<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(M_operator,fm);
}

// **********************************************************************
// 
// **********************************************************************
template<typename EvalT, typename Traits>
void MortarContact<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here
//
// Set-up moertel interface - only one for now. These are shared across worksets...
   const bool interface_is_oned = true;
   const int the_print_level = 0; 
   const int the_interface_index = 0;
   Teuchos::RCP<Epetra_Comm> moertel_comm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
   _moertelInterface = Teuchos::rcp(new MOERTEL::Interface(the_interface_index, interface_is_oned, *moertel_comm, the_print_level));

}

template<typename EvalT, typename Traits>
void MortarContact<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // We assume global search is done. Perform local search to pair up each master segment in the element
  // workset with the slave segments that it may potentially interact with

  // Then, form the mortar integration space


  // No work to do
  if(workset.sideSets == Teuchos::null || 
     this->masterSideNames.size() == 0 || 
     this->slaveSideNames.size() == 0 || 
     sideSetIDs.size() == 0)
    return;

  // currently only one pair of contact surfaces allowed.
  assert(masterSideNames.size()==1);
  assert(slaveSideNames.size()==1);

  const Albany::SideSetList& ssList = *(workset.sideSets);

  Albany::SideSetList::const_iterator it_master = ssList.find(masterSideNames[0]);
  Albany::SideSetList::const_iterator it_slave  = ssList.find(slaveSideNames[0]);

  const std::vector<Albany::SideStruct>& slaveSideSet = it_slave->second;
  const std::vector<Albany::SideStruct>& masterSideSet = it_master->second;

  int num_nodes = coordVec.dimension(1);
  std::size_t numFields = 2; //hack

  // If slave ss exists, loop over the slave sides and construct moertel nodes/faces and interface
  if(it_slave != ssList.end()) {
    std::set<int> inserted_nodes;
    std::set<int>::iterator it;
    std::pair<std::set<int>::iterator, bool> ret;
    for (std::size_t side=0; side < slaveSideSet.size(); ++side) {

      // Get the data that corresponds to the side. 
      // const int elem_GID   = slaveSideSet[side].elem_GID; // GID of the element that contains the master segment
      const int elem_LID   = slaveSideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
      // const int elem_side  = slaveSideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
      // const int elem_block = slaveSideSet[side].elem_ebIndex; // which  element block is the element in?

      // gather nodes from sideset and if unique then create moertel node
      const int  print_level = 4;     // experience from ALEGRA suggests this is a good choice... 
                                      // ... probably will want to parse this in production code
      const bool on_boundary = false; // will eventually want to allow boundaries to be intersected by contact surfaces
      const int  contact_pair_id = 0; // will eventually want to allow multiple pairs
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> wsElNodeID = workset.wsElNodeID;
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> &wsElNodeEqID = workset.wsElNodeEqID[elem_LID];
      for (std::size_t node=0; node < num_nodes; ++node) {
        ret = inserted_nodes.insert(wsElNodeID[elem_LID][node]);
        if (ret.second==true) { // this is a as yet unregistered node. add it
          const double coords[] = { coordVec(elem_LID, node, 0), coordVec(elem_LID, node, 1), 0.0 }; // Moertel node is 3 coords
          std::vector<int> list_of_dofgid;
          for (std::size_t eq=0; eq < numFields; eq++) {
            int global_eq_id = wsElNodeEqID[node][eq];
            list_of_dofgid.push_back(global_eq_id);
          }
          MOERTEL::Node moertel_node(wsElNodeID[elem_LID][node], 
                                     coords, list_of_dofgid.size(), 
                                     &list_of_dofgid[0], 
                                     on_boundary, 
                                     print_level);
          _moertelInterface->AddNode(moertel_node,contact_pair_id);
        }
      }
    }
  }
  
  if(it_master != ssList.end()) {
    for (std::size_t side=0; side < masterSideSet.size(); ++side) {
      //repeat slave side setup here.

    }
  }

/*
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
      }
*/


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

