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

  meshSpecs      (p.get<Teuchos::RCP<Albany::MeshSpecsStruct> >("Mesh Specs Struct")),
  // The array of names of all the master side sets in the problem
  masterSideNames (p.get<Teuchos::ArrayRCP<std::string> >("Master Sideset Names")), 

  // The array of sidesets to process
  sideSetIDs (p.get<Teuchos::ArrayRCP<std::string> >("Sideset IDs")), 

  // Node coords
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector) 

{
  std::string fieldName;

  // Allow the user to name this instance of the mortar projection operation
  // The default is just "MortarProjection"
  if (p.isType<std::string>("Projection Field Name"))
    fieldName = p.get<std::string>("Projection Field Name");
  else fieldName = "MortarProjection";

  mortar_projection_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  // Get the array of residual quantities that we need to project into the integration space
  // These are the physics residuals that this class evaluates
  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");

  tensorRank = p.get<int>("Tensor Rank");

  // scalar
  if (tensorRank == 0 ) {
    numFieldsBase = names.size();
    const std::size_t num_val = numFieldsBase;
    val.resize(num_val);
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      PHX::MDField<ScalarT,Cell,Node> mdf(names[eq],dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  }
  // vector
  else
  if (tensorRank == 1 ) {
    valVec.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec[0] = mdf;
    this->addDependentField(valVec[0]);
    numFieldsBase = dl->node_vector->dimension(2);
  }
  // tensor
  else
  if (tensorRank == 2 ) {
    valTensor.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim,Dim> mdf(names[0],dl->node_tensor);
    valTensor[0] = mdf;
    this->addDependentField(valTensor[0]);
    numFieldsBase = (dl->node_tensor->dimension(2))*(dl->node_tensor->dimension(3));
  }

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  // Tell PHAL which field we are evaluating. Note that this is actually a dummy, as we fill in the
  // residual vector directly. This tells PHAL to call this evaluator to satisfy this dummy field.
  this->addEvaluatedField(*mortar_projection_operation);

  this->setName(fieldName+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void MortarContact<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    numNodes = val[0].dimension(1);
  }
  else 
  if (tensorRank == 1) {
    this->utils.setFieldData(valVec[0],fm);
    numNodes = valVec[0].dimension(1);
  }
  else 
  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor[0],fm);
    numNodes = valTensor[0].dimension(1);
  }
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
  if(workset.sideSets == Teuchos::null || this->masterSideNames.size() == 0 || this->sideSetIDs.size() == 0)

    return;

  const Albany::SideSetList& ssList = *(workset.sideSets);

  for(std::size_t i = 0; i < this->sideSetIDs.size(); i++){

    Albany::SideSetList::const_iterator it = ssList.find(this->sideSetIDs[i]);

      if(it == ssList.end()) continue; // This sideset does not exist in this workset - try the next one

/*
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
       for (std::size_t node=0; node < numNodes; ++node)
         for (std::size_t dim=0; dim < 3; ++dim)
             neumann(cell, node, dim) = 0.0; // zero out the accumulation vector
*/

      const std::vector<Albany::SideStruct>& sideSet = it->second;

      // Loop over the sides that form the boundary condition

      for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

        // Get the data that corresponds to the side. 

        const int elem_GID = sideSet[side].elem_GID; // GID of the element that contains the master segment
        const int elem_LID = sideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
        const int elem_side = sideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
        const int elem_block = sideSet[side].elem_ebIndex; // which  element block is the element in?

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

