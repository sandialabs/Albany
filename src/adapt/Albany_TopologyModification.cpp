//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_TopologyModification.hpp"

#include "Albany_StressFracture.hpp"

#include "Teuchos_TimeMonitor.hpp"

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

static const double sqr(double v){ return v * v; }

Albany::TopologyMod::
TopologyMod(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
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

  double crit_stress = params->get<double>("Fracture Stress");

  sfcriterion = 
    Teuchos::rcp(new LCM::StressFracture(numDim, elementRank, avg_stresses, crit_stress, 
    *stk_discretization));

// Modified by GAH from LCM::NodeUpdate.cc

  topology =
    Teuchos::rcp(new LCM::topology(disc, sfcriterion));


}

Albany::TopologyMod::
~TopologyMod()
{
}

bool
Albany::TopologyMod::queryAdaptationCriteria(){


// FIXME Dumb criteria

   if(iter == 5 || iter == 10 || iter == 15){ // fracture at iter = 5, 10, 15

    // First, check and see if the mesh fracture criteria is met anywhere before messing with things.

    // Get a vector containing the edge set of the mesh where fractures can occur

    std::vector<stk::mesh::Entity*> edge_lst;
    stk::mesh::get_entities(*bulkData, numDim-1, edge_lst); // numDim - 1 is the edge (face in 3D) rank

    *out << "Num edges : " << edge_lst.size() << std::endl;

    // Probability that fracture_criterion will return true.
//    float p = 1.0;
    float p = iter;

    // Iterate over the boundary entities
    for (int i = 0; i < edge_lst.size(); ++i){

      stk::mesh::Entity& edge = *(edge_lst[i]);

      if(sfcriterion->fracture_criterion(edge, p))

        fractured_edges.push_back(edge_lst[i]);

    }

    if(fractured_edges.size() == 0) return false; // nothing to do

    *out << "TopologyModification: Need to split \"" 
         << fractured_edges.size() << "\" mesh elements." << std::endl;

    return true;

  }

  return false; 
 
}

bool
Albany::TopologyMod::adaptMesh(){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::TopologyMod method      " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

// FIXME - this needs to be made parallel!

  // Save the current element to node connectivity for solution transfer purposes

  buildElemToNodes(oldElemToNode);

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = baseExoFileName;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  std::cout << "Remeshing: renaming output file to - " << str << endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  remeshFileIndex++;

  Albany::StateArrays& currentState = StateMgr.getStateArrays();

  StateArray::iterator it;

  *out << "Found the following fields in the state manager:" << std::endl;
  for(it = currentState[0].begin(); it != currentState[0].end(); it++) // loop over all the arrays holding data
    *out << "\t" << it->first << std::endl; // print the name of the state


  int nWorksets = currentState.size();
  std::vector<int> dims;

// Make local storage to hold stress

//  stresses.resize(nWorksets);
//  for (int ws = 0; ws < nWorksets; ws++) {
//    currentState[ws]["Stress"].dimensions(dims);
//    stresses[ws].resize(dims);
//  }

  // Just use std::vector as we only store one double per cell in ws
  avg_stresses.resize(nWorksets);
  for (int ws = 0; ws < nWorksets; ws++) {
    currentState[ws]["Stress"].dimensions(dims);
    avg_stresses[ws].resize(dims[0]); // num cells long
  }

/*
// Copy stress over


  for (int ws = 0; ws < nWorksets; ws++)
  {
    currentState[ws]["Stress"].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 4 );
    
    for(int cell=0; cell < dims[0]; cell++)
      for(int qp=0; qp < dims[1]; qp++)
        for(int dim=0; dim < dims[2]; dim++)
          for(int dim2=0; dim2 < dims[3]; dim2++)
            stresses[ws](cell, qp, dim, dim2) = currentState[ws]["Stress"](cell, qp, dim, dim2);
  }
*/

// Copy stress over


/*
  for (int ws = 0; ws < nWorksets; ws++)
  {
    currentState[ws]["Stress"].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 4 );
    
    for(int cell=0; cell < dims[0]; cell++){
      avg_stresses[ws][cell] = 0;
      for(int qp=0; qp < dims[1]; qp++){
        double avg = 0;
        for(int dim=0; dim < dims[2]; dim++)
          avg += sqr(currentState[ws]["Stress"](cell, qp, dim, dim));
        avg = sqrt(avg);
        avg_stresses[ws][cell] += avg;
      }
      avg_stresses[ws][cell] /= (double)(dims[1] + 1);
   }
  }
*/

  for (int ws = 0; ws < nWorksets; ws++)
  {
    currentState[ws]["Stress"].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 4 );
    
    for(int cell=0; cell < dims[0]; cell++){
      avg_stresses[ws][cell] = 0;
      for(int qp=0; qp < dims[1]; qp++){
        avg_stresses[ws][cell] += currentState[ws]["Stress"](cell, qp, 1, 2);
      }
      avg_stresses[ws][cell] /= (double)(dims[1] + 1);
   }
  }

  // Print element connectivity before the mesh topology is modified

//  *out << "*************************\n"
//	   << "Before element separation\n"
//	   << "*************************\n";
  
  // Start the mesh update process

  // Modifies mesh for graph algorithm
  // Function must be called each time before there are changes to the mesh

  topology->remove_node_relations();

  // Check for failure criterion

  std::map<stk::mesh::EntityKey, bool> entity_open;

//  topology->set_entities_open(entity_open);
  topology->set_entities_open(fractured_edges, entity_open);

//cout << "Preparing to open " << entity_open.size() << " entities." << endl;

//  std::string gviz_output = "output.dot";
//  topology.output_to_graphviz(gviz_output,entity_open);

  // test the functions of the class

  bulkData->modification_begin();

  // begin mesh fracture

//  *out << "begin mesh fracture\n";

  topology->fracture_boundary(entity_open);

  // Clear the list of fractured edges in preparation for the next fracture event

  fractured_edges.clear();

  //std::string gviz_output = "output.dot";
  //topology.output_to_graphviz(gviz_output,entity_open);

  // Recreates connectivity in stk mesh expected by Albany_STKDiscretization
  // Must be called each time at conclusion of mesh modification

  topology->graph_cleanup();

  // End mesh update

  bulkData->modification_end();

//  *out << "*************************\n"
//	   << "After element separation\n"
//	   << "*************************\n";

//  topology->disp_connectivity();

  // Throw away all the Albany data structures and re-build them from the mesh

  stk_discretization->updateMesh();

  // Save the new element to node connectivity for solution transfer purposes

  buildElemToNodes(newElemToNode);

  return true;

}

//! Transfer solution between meshes.
void
Albany::TopologyMod::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

  // Note: This code assumes that the number of elements and their relationships between the old and
  // new meshes do not change!

  // Logic: When we split an edge(s), the number of elements are unchanged. On the two elements that share
  // the split edge, the node numbers along the split edge change, as does the location of the "physics"
  // in the solution vector for these nodes. Here, we loop over the elements in the old mesh, and copy the
  // "physics" at the nodes to the proper locations for the element's new nodes in the new mesh.

// FIXME clean this up and make it parallel-aware

  int neq = (disc->getWsElNodeEqID())[0][0][0].size();

  for(int elem = 0; elem < oldElemToNode.size(); elem++)

    for(int node = 0; node < oldElemToNode[elem].size(); node++){

       int onode = oldElemToNode[elem][node];
       int nnode = newElemToNode[elem][node];

       for(int eq = 0; eq < neq; eq++)

//          newSolution[disc->getOwnedDOF(nnode, eq)] =
//             oldSolution[disc->getOwnedDOF(onode, eq)];
          newSolution[newElemToNode[elem][node] * neq + eq] =
             oldSolution[oldElemToNode[elem][node] * neq + eq];

    }

}



Teuchos::RCP<const Teuchos::ParameterList>
Albany::TopologyMod::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidTopologyModificationParams");

/*
  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
*/

  validPL->set<double>("Fracture Stress", 1.0, "Fracture stress value at which two elements separate");

  return validPL;
}

void
Albany::TopologyMod::showElemToNodes(){

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

void
Albany::TopologyMod::buildElemToNodes(std::vector<std::vector<int> >& connectivity){

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

void
Albany::TopologyMod::showRelations(){

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


