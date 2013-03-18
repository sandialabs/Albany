//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RandomFracture.hpp"

#include "Albany_RandomCriterion.hpp"

#include "Teuchos_TimeMonitor.hpp"

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

Albany::RandomFracture::
RandomFracture(const Teuchos::RCP<Teuchos::ParameterList>& params_,
               const Teuchos::RCP<ParamLib>& paramLib_,
               Albany::StateManager& StateMgr_,
               const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
  remeshFileIndex(1),
  fracture_interval_(params_->get<int>("Adaptivity Step Interval", 1)),
  fracture_probability_(params_->get<double>("Fracture Probability", 1.0))
{

  disc = StateMgr.getDiscretization();

  stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

  stkMeshStruct = stk_discretization->getSTKMeshStruct();

  bulkData = stkMeshStruct->bulkData;
  metaData = stkMeshStruct->metaData;

  // The entity ranks
  nodeRank = metaData->NODE_RANK;
  edgeRank = metaData->EDGE_RANK;
  faceRank = metaData->FACE_RANK;
  elementRank = metaData->element_rank();

  numDim = stkMeshStruct->numDim;

  // Save the initial output file name
  baseExoFileName = stkMeshStruct->exoOutFile;

  sfcriterion = 
    Teuchos::rcp(new LCM::RandomCriterion(numDim, 
                                          elementRank,
                                          *stk_discretization));

  // Modified by GAH from LCM::NodeUpdate.cc
  topology =
    Teuchos::rcp(new LCM::topology(disc, sfcriterion));


}

Albany::RandomFracture::
~RandomFracture()
{
}

bool
Albany::RandomFracture::queryAdaptationCriteria()
{
  *out << " HELP: in RandomFracture::query..." << endl;
  *out << " HELP: iter: " << iter << endl;
  *out << " HELP: fracture_interval: " << fracture_interval_ << endl;
  *out << " HELP: iter % fracture_interval: " << iter % fracture_interval_ << endl;
  // iter is a member variable elsewhere, NOX::Epetra::AdaptManager.H
  if ( iter % fracture_interval_ == 0) {

    // Get a vector containing the face set of the mesh where
    // fractures can occur
    std::vector<stk::mesh::Entity*> face_list;
    stk::mesh::get_entities(*bulkData, numDim-1, face_list);

#ifdef ALBANY_VERBOSE
    std::cout << "Num faces : " << face_list.size() << std::endl;
#endif

    // keep count of total fractured faces
    int total_fractured;

    *out << " HELP: fracture_probability: " << fracture_probability_ << std::endl;

    // Iterate over the boundary entities
    for (int i = 0; i < face_list.size(); ++i){

      stk::mesh::Entity& face = *(face_list[i]);

      if(sfcriterion->fracture_criterion(face, fracture_probability_)) {
        *out << "   HELP: face fractured: " << i << std::endl;
        fractured_edges.push_back(face_list[i]);
      }
    }

    // if(fractured_edges.size() == 0) return false; // nothing to do
    if((total_fractured = accumulateFractured(fractured_edges.size())) == 0) {

      fractured_edges.clear();

      return false; // nothing to do

    }


    std::cout << "RandomFractureification: Need to split \"" 
      //    *out << "RandomFractureification: Need to split \"" 
              << total_fractured << "\" mesh elements." << std::endl;


    return true;

  }

  return false; 
 
}

bool
Albany::RandomFracture::adaptMesh(){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::RandomFracture method   " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // Save the current element to node connectivity for solution
  // transfer purposes

  oldElemToNode.clear();
  newElemToNode.clear();

  // buildElemToNodes(oldElemToNode);

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the
  // remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = baseExoFileName;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  std::cout << "Remeshing: renaming output file to - " << str << endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  // increment name index
  remeshFileIndex++;

  // perform topology operations
  topology->remove_element_to_node_relations(oldElemToNode);

  // Check for failure criterion
  std::map<stk::mesh::EntityKey, bool> entity_open;
  topology->set_entities_open(fractured_edges, entity_open);

  // begin mesh update
  bulkData->modification_begin();

  // FIXME parallel bug lies in here
  topology->fracture_boundary(entity_open, oldElemToNode, newElemToNode);

  // Clear the list of fractured edges in preparation for the next
  // fracture event
  fractured_edges.clear();

  // Recreates connectivity in stk mesh expected by
  // Albany_STKDiscretization Must be called each time at conclusion
  // of mesh modification
  topology->restore_element_to_node_relations(newElemToNode);

  // end mesh update
  bulkData->modification_end();

  // Throw away all the Albany data structures and re-build them from
  // the mesh
  stk_discretization->updateMesh();

  return true;

}

//! Transfer solution between meshes.
void
Albany::RandomFracture::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution){

  // Note: This code assumes that the number of elements and their
  // relationships between the old and new meshes do not change!

  // Logic: When we split an edge(s), the number of elements are
  // unchanged. On the two elements that share the split edge, the
  // node numbers along the split edge change, as does the location of
  // the "physics" in the solution vector for these nodes. Here, we
  // loop over the elements in the old mesh, and copy the "physics" at
  // the nodes to the proper locations for the element's new nodes in
  // the new mesh.

  int neq = (disc->getWsElNodeEqID())[0][0][0].size();

  for(int elem = 0; elem < oldElemToNode.size(); elem++) {

    for(int node = 0; node < oldElemToNode[elem].size(); node++) {

      int onode = oldElemToNode[elem][node]->identifier() - 1;
      int nnode = newElemToNode[elem][node]->identifier() - 1;

      for(int eq = 0; eq < neq; eq++) {
        
        newSolution[nnode * neq + eq] =
          oldSolution[onode * neq + eq];
      }
    }
  }
}



Teuchos::RCP<const Teuchos::ParameterList>
Albany::RandomFracture::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidRandomFractureificationParams");

  validPL->set<double>("Fracture Probability", 
                       1.0, 
                       "Probability of fracture");
  validPL->set<double>("Adaptivity Step Interval", 
                       1, 
                       "Interval to check for fracture");

  return validPL;
}

void
Albany::RandomFracture::showElemToNodes(){

  // Create a list of element entities
  std::vector<Entity*> element_lst;
  stk::mesh::get_entities(*(bulkData), elementRank, element_lst);

  // Loop over the elements
  for (int i = 0; i < element_lst.size(); ++i){
    stk::mesh::PairIterRelation relations = element_lst[i]->relations(nodeRank);
    cout << "Nodes of Element " << element_lst[i]->identifier() - 1 << "\n";

    for (int j = 0; j < relations.size(); ++j){
      Entity& node = *(relations[j].entity());
      cout << ":"  << node.identifier() - 1;
    }
    cout << ":\n";
  }

  //topology::disp_relation(*(element_lst[0]));

  //std::vector<Entity*> face_lst;
  //stk::mesh::get_entities(*(bulkData_),elementRank-1,face_lst);
  //topology::disp_relation(*(face_lst[1]));

  return;
}

/*
  void
  Albany::RandomFracture::buildElemToNodes(std::vector<std::vector<int> >& connectivity){

  // Create a list of element entities
  std::vector<Entity*> element_lst;
  stk::mesh::get_entities(*(bulkData), elementRank, element_lst);

  // Allocate storage for the elements

  connectivity.resize(element_lst.size());

  // Loop over the elements
  for (int i = 0; i < element_lst.size(); ++i){
  stk::mesh::PairIterRelation relations = element_lst[i]->relations(nodeRank);
  int element = element_lst[i]->identifier() - 1;

  // make room to hold the node ids
  connectivity[element].resize(relations.size());

  for (int j = 0; j < relations.size(); ++j){
  Entity& node = *(relations[j].entity());
  connectivity[element][j] = node.identifier() - 1;
  }
  }
  return;
  }
*/

void
Albany::RandomFracture::showRelations(){

  std::vector<Entity*> element_lst;
  stk::mesh::get_entities(*(bulkData),elementRank,element_lst);

  // Remove extra relations from element
  for (int i = 0; i < element_lst.size(); ++i){
    Entity & element = *(element_lst[i]);
    stk::mesh::PairIterRelation relations = element.relations();
    std::cout << "Element " << element_lst[i]->identifier() << " relations are :" << std::endl;

    for (int j = 0; j < relations.size(); ++j){
      cout << "entity:\t" << relations[j].entity()->identifier() << ","
           << relations[j].entity()->entity_rank() << "\tlocal id: "
           << relations[j].identifier() << "\n";
    }
  }
}


#ifdef ALBANY_MPI
int
Albany::RandomFracture::accumulateFractured(int num_fractured){

  int total_fractured;
  const Albany_MPI_Comm& mpi_comm = Albany::getMpiCommFromEpetraComm(*comm);

  MPI_Allreduce(&num_fractured, &total_fractured, 1, MPI_INT, MPI_SUM, mpi_comm);

  return total_fractured;
}
#else
int
Albany::RandomFracture::accumulateFractured(int num_fractured){
  return num_fractured;
}
#endif

