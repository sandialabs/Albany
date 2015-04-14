//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"
#include "PeridigmManager.hpp"
#include "Albany_AbstractDiscretization.hpp"

template<typename EvalT, typename Traits>
PHAL::AlbanyPeridigmOBCFunctional<EvalT, Traits>::
AlbanyPeridigmOBCFunctional(Teuchos::ParameterList& p,
			    const Teuchos::RCP<Albany::Layouts>& dl) :
  referenceCoordinates ("Coord Vec", dl->vertices_vector),
  displacement         ("Displacement", dl->node_vector)
{
//   Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");

  std::cout << "OBC DEBUGGING AlbanyPeridigmOBCFunctional::AlbanyPeridigmOBCFunctional()" << std::endl;

  // add dependent fields
  this->addDependentField(referenceCoordinates);
  this->addDependentField(displacement);
  this->setName("OBC Functional"+PHX::typeAsString<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  PHX::Tag<ScalarT> local_response_tag("Local OBC Functional", dl->node_scalar);
  PHX::Tag<ScalarT> global_response_tag("Global OBC Functional", dl->workset_scalar);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::AlbanyPeridigmOBCFunctional<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(referenceCoordinates,fm);
  this->utils.setFieldData(displacement,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::AlbanyPeridigmOBCFunctional<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response, 0.0);
  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::AlbanyPeridigmOBCFunctional<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::cout << "OBC DEBUGGING AlbanyPeridigmOBCFunctional::evaluateFields() top of function" << std::endl;

  LCM::PeridigmManager& peridigmManager = LCM::PeridigmManager::self();

  // Zero out local response
  PHAL::set(this->local_response, 0.0);

  Teuchos::RCP<Albany::STKDiscretization> stkDisc = peridigmManager.getSTKDisc();
  TEUCHOS_TEST_FOR_EXCEPT_MSG(stkDisc.is_null(), "\n\n**** Error in PeridigmManager::initialize():  stkDisc.is_null() == true.\n\n");
  Teuchos::RCP<const stk::mesh::BulkData> bulkData = Teuchos::rcpFromRef(stkDisc->getSTKBulkData());

  // WsLIDList is a std::map<GO, wsLid>
  // The key is the global id and the value is the workset local id
  // wsLid is a struct containing ws (workset id), and LID (local id of the element)
  Albany::WsLIDList& elemIdData = stkDisc->getElemGIDws();
  for(Albany::WsLIDList::iterator it=elemIdData.begin() ; it!=elemIdData.end() ; it++){
    GO globalElemId = it->first;
    int worksetId = it->second.ws;
    int worksetElemId = it->second.LID;

  }

  // Global solution vector, which contains nodal displacements
  Teuchos::RCP<const Epetra_Vector> displacement = workset.x;

  Teuchos::RCP< std::vector<LCM::PeridigmManager::OBCDataPoint> > obcDataPoints = peridigmManager.getOBCDataPoints();

  int numCells = 1;
  int numPoints = 1;
  int numDim = 3;

  Intrepid::FieldContainer<RealType> physPoints;
  physPoints.resize(numCells, numPoints, numDim);

  Intrepid::FieldContainer<RealType> refPoints;
  refPoints.resize(numCells, numPoints, numDim);

  for(unsigned int iEvalPt=0 ; iEvalPt<obcDataPoints->size() ; iEvalPt++){

    for(int dof=0 ; dof<3 ; dof++){
      refPoints(0, 0, dof) = (*obcDataPoints)[iEvalPt].naturalCoords[dof];
    }

    stk::mesh::Entity elem = (*obcDataPoints)[iEvalPt].albanyElement;
    int globalElemId = bulkData->identifier(elem) - 1;

    Albany::WsLIDList::iterator it = elemIdData.find(globalElemId);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(it == elemIdData.end(), "Error in AlbanyPeridigmOBCFunctional evalutor, failed to find albany element.");
    int worksetId = it->second.ws;
    if(worksetId == workset.wsIndex){
      std::cout << "DJL in workset" << std::endl;
    }
    else{
      std::cout << "DJL not in workset" << std::endl;
    }

//     int numNodes = bulkData->num_nodes((*obcDataPoints)[iEvalPt].albanyElement);
//     const stk::mesh::Entity* nodes = bulkData->begin_nodes((*obcDataPoints)[iEvalPt].albanyElement);

//     Intrepid::FieldContainer<RealType> cellWorkset;
//     cellWorkset.resize(numCells, numNodes, numDim);
//     for(int i=0 ; i<numNodes ; i++){
//       int globalAlbanyNodeId = bulkData->identifier(nodes[i]) - 1;
//       Tpetra_Map::local_ordinal_type albanyLocalId = albanyMap->getLocalElement(3*globalAlbanyNodeId);
//       TEUCHOS_TEST_FOR_EXCEPT_MSG(albanyLocalId == Teuchos::OrdinalTraits<LO>::invalid(), "\n\n**** Error in PeridigmManager::obcEvaluateFunctional(), invalid Albany local id.\n\n");
//       cellWorkset(0, i, 0) = albanyCurrentDisplacement[albanyLocalId];
//       cellWorkset(0, i, 1) = albanyCurrentDisplacement[albanyLocalId + 1];
//       cellWorkset(0, i, 2) = albanyCurrentDisplacement[albanyLocalId + 2];
//     }

//     shards::CellTopology cellTopology(&(*obcDataPoints)[iEvalPt].cellTopologyData);

//     Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, cellTopology);

//     // Record the difference between the Albany displacement at the point (which was just computed using Intrepid) and
//     // the Peridigm displacement at the point
//     for(int dof=0 ; dof<3 ; dof++){
//       displacementDiff[3*iEvalPt+dof] = physPoints(0,0,dof) - ((*obcDataPoints)[iEvalPt].currentCoords[dof] - (*obcDataPoints)[iEvalPt].initialCoords[dof]);
//     }
  }










//   ScalarT s;
//   for (std::size_t cell=0; cell < workset.numCells; ++cell) {

//     for (std::size_t qp=0; qp < numQPs; ++qp) {
//       if (field_rank == 2) {
// 	s = field(cell,qp) * weights(cell,qp) * scaling;
// 	this->local_response(cell,0) += s;
// 	this->global_response(0) += s;
//       }
//       else if (field_rank == 3) {
// 	for (std::size_t dim=0; dim < field_components.size(); ++dim) {
// 	  s = field(cell,qp,field_components[dim]) * weights(cell,qp) * scaling;
// 	  this->local_response(cell,dim) += s;
// 	  this->global_response(dim) += s;
// 	}
//       }
//       else if (field_rank == 4) {
// 	for (std::size_t dim1=0; dim1 < field_dims[2]; ++dim1) {
// 	  for (std::size_t dim2=0; dim2 < field_dims[3]; ++dim2) {
// 	    s = field(cell,qp,dim1,dim2) * weights(cell,qp) * scaling;
// 	    this->local_response(cell,dim1,dim2) += s;
// 	    this->global_response(dim1,dim2) += s;
// 	  }
// 	}
//       }
//     }
//   }

  std::cout << "OBC DEBUGGING AlbanyPeridigmOBCFunctional::evaluateFields() about to call PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset)" << std::endl;

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);

  std::cout << "OBC DEBUGGING AlbanyPeridigmOBCFunctional::evaluateFields() bottom of function" << std::endl;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::AlbanyPeridigmOBCFunctional<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
PHAL::AlbanyPeridigmOBCFunctional<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid AlbanyPeridigmOBCFunctional Params"));
//   Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
//     PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
//   validPL->setParameters(*baseValidPL);

//   validPL->set<std::string>("Name", "", "Name of response function");
//   validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
//   validPL->set<std::string>("Field Type", "", "Type of field (scalar, vector, ...)");
//   validPL->set<std::string>(
//     "Element Block Name", "", 
//     "Name of the element block to use as the integration domain");
//   validPL->set<std::string>("Field Name", "", "Field to integrate");
//   validPL->set<bool>("Positive Return Only",false);

//   validPL->set<double>("Length Scaling", 1.0, "Length Scaling");
//   validPL->set<double>("x min", 0.0, "Integration domain minimum x coordinate");
//   validPL->set<double>("x max", 0.0, "Integration domain maximum x coordinate");
//   validPL->set<double>("y min", 0.0, "Integration domain minimum y coordinate");
//   validPL->set<double>("y max", 0.0, "Integration domain maximum y coordinate");
//   validPL->set<double>("z min", 0.0, "Integration domain minimum z coordinate");
//   validPL->set<double>("z max", 0.0, "Integration domain maximum z coordinate");

//   validPL->set< Teuchos::Array<int> >("Field Components", Teuchos::Array<int>(),
// 				      "Field components to scatter");

  return validPL;
}

// **********************************************************************

