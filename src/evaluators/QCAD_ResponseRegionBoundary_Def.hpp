/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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

  // User-specified parameters
  regionType     = plist->get<std::string>("Region Type");  // "Element Blocks", "Quantum Blocks", "Boxed Level Set"
  outputFilename = plist->get<std::string>("Output Filename");

  //Always check for Element Block Names param (not just for "Element Blocks" region type
  std::string ebNameStr = plist->get<std::string>("Element Block Names","");
  if(ebNameStr.length() > 0) Albany::splitStringOnDelim(ebNameStr,',',ebNames);

  bQuantumEBsOnly = plist->get<bool>("Quantum Element Blocks Only",false);

  if(regionType == "Quantum Blocks") {
    bQuantumEBsOnly = true;
  }
  else if(regionType == "Boxed Level Set") {
    levelSetFieldname = plist->get<std::string>("Level Set Field Name");
    levelSetFieldMin = plist->get<double>("Level Set Field Minimum", -1e100);
    levelSetFieldMax = plist->get<double>("Level Set Field Maximum", +1e100);

    PHX::MDField<ScalarT> f(levelSetFieldname, scalar_dl); 
    levelSetField = f;
    this->addDependentField(levelSetField);
  }

  // add dependent fields
  this->addDependentField(coordVec);
  this->addDependentField(weights);

  // Create field tag: NOTE: may have name conflicts here: TODO: create a *unique* name based on the parameters (EGN)
  response_field_tag = Teuchos::rcp(new PHX::Tag<ScalarT>(regionType + " Get Region Boundary Response",
							  dl->dummy));
  this->addEvaluatedField(*response_field_tag);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseRegionBoundary<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if(regionType == "Boxed Level Set") this->utils.setFieldData(levelSetField,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseRegionBoundary<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT avgCellVal, cellVol;
  bool bQuantumEB = false; //if no material database, all element blocks are "non-quantum"

  if(materialDB != Teuchos::null)
    bQuantumEB = materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false);

  //check if this element block should be considered
  if( (ebNames.size() == 0 || 
       std::find(ebNames.begin(), ebNames.end(), workset.EBName) != ebNames.end()) &&
      (bQuantumEBsOnly == false || bQuantumEB == true) ) {

    if(regionType == "Boxed Level Set") {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

	// Get the cell volume, used for averaging over a cell
	cellVol = 0.0;
	for (std::size_t qp=0; qp < numQPs; ++qp)
	  cellVol += weights(cell,qp);

	// Get the average value for the cell (integral of field over cell divided by cell volume)
	avgCellVal = 0.0;
	for (std::size_t qp=0; qp < numQPs; ++qp) {
	  avgCellVal += levelSetField(cell,qp) * weights(cell,qp);
	}
	avgCellVal /= cellVol;

	if( avgCellVal <= levelSetFieldMax && avgCellVal >= levelSetFieldMin) {
	  
	  //Update min/max values using coordinates of quad points in this cell - better way using nodes?
	  for (std::size_t qp=0; qp < numQPs; ++qp)  {
	    for(std::size_t i=0; i<numDims; i++) {
	      if(minVals[i] > coordVec(cell,qp,i)) minVals[i] = coordVec(cell,qp,i);
	      if(maxVals[i] < coordVec(cell,qp,i)) maxVals[i] = coordVec(cell,qp,i);
	    }
	  }

	}
      }
    }
    else {  //other than "Boxed Level Set"
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
	
	//Update min/max values using coordinates of quad points in this cell - better way using nodes?
	for (std::size_t qp=0; qp < numQPs; ++qp)  {
	  for(std::size_t i=0; i<numDims; i++) {
	    if(minVals[i] > coordVec(cell,qp,i)) minVals[i] = coordVec(cell,qp,i);
	    if(maxVals[i] < coordVec(cell,qp,i)) maxVals[i] = coordVec(cell,qp,i);
	  }
	}
      }
    }
  } // end check if element block should be considered
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
 
  //Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MeshScalarT> > serializer =
  //  workset.serializerManager.template getValue<EvalT>();

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

  validPL->set<string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Region Type", "Element Blocks", "How region is defined: 'Element Blocks', 'Quantum Blocks', 'Boxed Level Set'");
  validPL->set<string>("Output Filename", "<filename>", "The file to write region boundary (min/max values) to");
  validPL->set<string>("Element Block Names", "", "Names of element blocks to consider, comma delimited");
  validPL->set<bool>("Quantum Element Blocks Only", false);
  validPL->set<string>("Level Set Field Name", "<field name>",
		       "Scalar Field to use for 'Boxed Level Set' region type");
  validPL->set<double>("Level Set Field Minimum", 0.0, "Minimum value of field to include in region when using Boxed Level Set type");
  validPL->set<double>("Level Set Field Maximum", 0.0, "Maximum value of field to include in region when using Boxed Level Set type");

  validPL->set<string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

