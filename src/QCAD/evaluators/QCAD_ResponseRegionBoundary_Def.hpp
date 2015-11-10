//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Albany_Utils.hpp"

template<typename EvalT, typename Traits>
QCAD::ResponseRegionBoundary<EvalT, Traits>::
ResponseRegionBoundary(Teuchos::ParameterList& p,
		   const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_vector),
  weights("Weights", dl->qp_scalar)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! parameters passed down from problem
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem = 
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

    // Material database (if given)
  if(paramsFromProblem != Teuchos::null)
    materialDB = paramsFromProblem->get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
  else materialDB = Teuchos::null;

  // number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  minVals.resize(numDims);
  maxVals.resize(numDims);
  for(std::size_t i=0; i<numDims; i++) {
    minVals[i] =  1e100;
    maxVals[i] = -1e100;
  }

  //! initialize operation region / domain
  opRegion  = Teuchos::rcp( new QCAD::MeshRegion<EvalT, Traits>("Coord Vec","Weights",*plist,materialDB,dl) );

  // User-specified parameters
  outputFilename = plist->get<std::string>("Output Filename");

  // add dependent fields
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  opRegion->addDependentFields(this);

  response_field_tag = Teuchos::rcp(new PHX::Tag<ScalarT>("Get Region Boundary Response -> " + outputFilename,
							  dl->dummy));
  this->addEvaluatedField(*response_field_tag);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseRegionBoundary<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  opRegion->postRegistrationSetup(fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseRegionBoundary<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if(!opRegion->elementBlockIsInRegion(workset.EBName))
    return;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    if(!opRegion->cellIsInRegion(cell)) continue;
    
    //Update min/max values using coordinates of quad points in this cell - better way using nodes?
    for (std::size_t qp=0; qp < numQPs; ++qp)  {
      for(std::size_t i=0; i<numDims; i++) {
	if(minVals[i] > coordVec(cell,qp,i)) minVals[i] = coordVec(cell,qp,i);
	if(maxVals[i] < coordVec(cell,qp,i)) maxVals[i] = coordVec(cell,qp,i);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseRegionBoundary<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  //Only perform this post-evaluation for the residual case or jacobian cases, 
  // since we just write coordinate values to an output file.  In particular, Tangent doesn't
  // work with EvalUatorTools correctly yet, so we want to skip it for now.
  if(QCAD::EvaluatorTools<EvalT,Traits>::getEvalType() != "Residual" && 
     QCAD::EvaluatorTools<EvalT,Traits>::getEvalType() != "Jacobian") {
    return;
  }

  std::vector<double> global_minVals(numDims), global_maxVals(numDims);

  // EGN: Removed (commented out) serializer since this gives templating compile errors
  //   for MeshScalarT.
 
  for(std::size_t i=0; i<numDims; i++) {
    // Compute contributions across processors
    //Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_MIN, 1, 
    //                    &global_minVals[i], &minVals[i]);
    //Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_MAX, 1,
    //                    &global_maxVals[i], &maxVals[i]);
    double minVal = QCAD::EvaluatorTools<EvalT,Traits>::getMeshDoubleValue(minVals[i]);
    double maxVal = QCAD::EvaluatorTools<EvalT,Traits>::getMeshDoubleValue(maxVals[i]);
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_MIN, 1, &minVal, &(global_minVals[i]));
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_MAX, 1, &maxVal, &(global_maxVals[i]));

    //DEBUG (assumes 2D case)
    //std::cout << "DEBUG: min/max vals["<<i<<"] (global) = " << global_minVals[i] << "," << global_maxVals[i] << std::endl;

    //Note: we really don't need any of this fancy MPI to broadcast the global min/max to each processor
    //  since only rank 0 writes the output file, but keep here for reference.
    /*int procToBcast_max = -1, procToBcast_min = -1;
    if( global_minVals[i] == minVal ) 
      procToBcast_min = workset.comm->getRank();
    if( global_maxVals[i] == maxVal ) 
      procToBcast_max = workset.comm->getRank();

    int winner_max, winner_min;
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_MAX, 1, &procToBcast_min, &winner_min);
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_MAX, 1, &procToBcast_max, &winner_max);

    std::cout << "DEBUG: min/max winners = " << winner_min << ", " << winner_max << std::endl;
    //Teuchos::broadcast(*workset.comm, *serializer, winner_min, 1, &global_minVals[i]);
    //Teuchos::broadcast(*workset.comm, *serializer, winner_max, 1, &global_maxVals[i]);
    Teuchos::broadcast(*workset.comm, winner_min, 1, &global_minVals[i]);
    Teuchos::broadcast(*workset.comm, winner_max, 1, &global_maxVals[i]);
    */
  }

  //Now global_minVals and global_maxVals have global min/max coordinate values on each proc
  
  // Rank 0 proc outputs to file
  if(workset.comm->getRank() == 0) {
    if( outputFilename.length() > 0) {
      std::fstream out;
      out.open(outputFilename.c_str(), std::fstream::out);
      out << "# i-th line below gives <min> <max> values for i-th dimension" << std::endl;
      
      for(std::size_t i=0; i<numDims; i++)
	out << global_minVals[i] << " " << global_maxVals[i] << std::endl;

      out.close();
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseRegionBoundary<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseRegionBoundary Params"));

  Teuchos::RCP<const Teuchos::ParameterList> regionValidPL =
    QCAD::MeshRegion<EvalT,Traits>::getValidParameters();
  validPL->setParameters(*regionValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Output Filename", "<filename>", "The file to write region boundary (min/max values) to");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

