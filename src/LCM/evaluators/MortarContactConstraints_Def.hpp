//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Constructor
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
MortarContact<EvalT, Traits>::
MortarContact(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :

  meshSpecs      (p.get<const Albany::MeshSpecsStruct*>("Mesh Specs Struct")),
  // The array of names of all the master side sets in the problem
  masterSideNames (p.get<Teuchos::Array<std::string> >("Master Side Set Names")), 
  slaveSideNames (p.get<Teuchos::Array<std::string> >("Slave Side Set Names")), 

  // The array of sidesets to process
  sideSetIDs (p.get<Teuchos::Array<std::string> >("Sideset IDs")), 

  // Node coords
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),

  // Fill in M and D in this evaluator
  M_operator       (p.get<std::string>("M Name"), dl->qp_scalar) 

{

  // Print the master side set names
  std::cout << "Master side sets found, number = " << masterSideNames.size() << std::endl;
  for(int i = 0; i < masterSideNames.size(); i++)
    std::cout << masterSideNames[i] << std::endl;

  // Print the slave side set names
  std::cout << "Slave side sets found, number = " << slaveSideNames.size() << std::endl;
  for(int i = 0; i < slaveSideNames.size(); i++)
    std::cout << slaveSideNames[i] << std::endl;

  // Print all sideset ids
  for(int i = 0; i < sideSetIDs.size(); i++)
    std::cout << sideSetIDs[i] << std::endl;

  // This evaluator uses the nodal coordinates to form the M and D operator
  this->addDependentField(coordVec);
  this->addEvaluatedField(M_operator);

  this->setName("Mortar Contact Constraints"+PHX::TypeString<EvalT>::value);
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
     this->slaveSideNames.size() ==0 || 
     sideSetIDs.size() == 0)
    return;

  const Albany::SideSetList& ssList = *(workset.sideSets);

  for(std::size_t i = 0; i < sideSetIDs.size(); i++){

std::cout << "The sideset ID for sideset : " << i << " is : " << sideSetIDs[i] << std::endl;

    Albany::SideSetList::const_iterator it = ssList.find(sideSetIDs[i]);

      if(it == ssList.end()) continue; // This sideset does not exist in this workset - try the next one

/*
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
       for (std::size_t node=0; node < numNodes; ++node)
         for (std::size_t dim=0; dim < 3; ++dim)
             neumann(cell, node, dim) = 0.0; // zero out the accumulation vector
*/

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



  // Then assemble the DOFs (flux, traction) at the slaves into the master side local elements

#if 0  // Here is the assemble code, more or less

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->val[eq])(cell,node);
    }
  } else 
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->valVec[0])(cell,node,eq);
    }
  } else
  if (this->tensorRank == 2) {
    int numDims = this->valTensor[0].dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t i = 0; i < numDims; i++)
          for (std::size_t j = 0; j < numDims; j++)
            f_nonconstView[nodeID[node][this->offset + i*numDims + j]] += (this->valTensor[0])(cell,node,i,j);
  
    }
  }
#endif
}


}

