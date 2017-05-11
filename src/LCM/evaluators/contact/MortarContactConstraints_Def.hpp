//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Utils.hpp"

// **********************************************************************
// Constructor
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
MortarContactBase<EvalT, Traits>::
MortarContactBase(Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec          (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector), //Node coords
  M_operator        (p.get<std::string>("M Name"), dl->qp_scalar)  //M portion of G

{

std::cout << "Calling MortarContactConstraintsBase constructor in " << __FILE__ << " line " << __LINE__ << " . " << std::endl;

  // This evaluator uses the nodal coordinates to form the M and D operator
  this->addDependentField(coordVec);
  this->addEvaluatedField(M_operator);

  this->setName("Mortar Contact Constraints"+PHX::typeAsString<EvalT>());

}

// **********************************************************************
template<typename EvalT, typename Traits>
void MortarContactBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
std::cout << "Calling ContactConstraints postRegistrationSetup in " << __FILE__ << " line " << __LINE__ << " . " << std::endl;
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(M_operator, fm);
}

// **********************************************************************
// 
// **********************************************************************
// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************

template<typename Traits>
MortarContact<PHAL::AlbanyTraits::Residual, Traits>::
MortarContact(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      MortarContactBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl){
std::cout << "Calling ContactConstraints constructor in " << __FILE__ << " line " << __LINE__ << " . " << std::endl;
}


template<typename Traits>
void MortarContact<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

std::cout << "Calling ContactConstraints evaluateFields in " << __FILE__ << " line " << __LINE__ << " . " << std::endl;


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

