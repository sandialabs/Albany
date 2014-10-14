//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"


template<typename EvalT, typename Traits>
QCAD::ResponseSaddleValue<EvalT, Traits>::
ResponseSaddleValue(Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("Coordinate Vector Name"), dl->qp_vector),
  coordVec_vertices(p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),
  weights(p.get<std::string>("Weights Name"), dl->qp_scalar)
{
  using Teuchos::RCP;
  
  //! get lattice temperature and materialDB from "Parameters From Problem"
  RCP<Teuchos::ParameterList> probList = 
    p.get< RCP<Teuchos::ParameterList> >("Parameters From Problem");
  lattTemp = probList->get<double>("Temperature");
  materialDB = probList->get< RCP<QCAD::MaterialDatabase> >("MaterialDB");
  
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! get pointer to response function object
  svResponseFn = plist->get<Teuchos::RCP<QCAD::SaddleValueResponseFunction> >
    ("Response Function");

  //! number of quad points per cell and dimension of space
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[2];

  //! User-specified parameters
  fieldName  = plist->get<std::string>("Field Name");
  
  // limit to Potential only because other fields such as CB show large error 
  // and very jaggy profile, which may be due to averaging effect because Ec is not
  // well defined at the Si/SiO2 interface (discontinuous), while Potential is always continuous.
  if (fieldName != "Potential") 
     TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
		      << "Error! Field Name must be Potential" << std::endl); 

  fieldGradientName  = plist->get<std::string>("Field Gradient Name");
  scaling = plist->get<double>("Field Scaling Factor",-1.0);
  gradScaling = plist->get<double>("Field Gradient Scaling Factor",-1.0);

  retFieldName = plist->get<std::string>("Return Field Name", fieldName);
  retScaling = plist->get<double>("Return Field Scaling Factor",1.0);
  bReturnSameField = (fieldName == retFieldName);
  //bLateralVolumes = true; // Future: make into a parameter

  //! Special case when return field name == "current": then just compute 
  //   as if returning the same field, and overwrite with current value at end
  if(retFieldName == "current")
    bReturnSameField = true;    

  //! setup operation field and its gradient, and the return field (if it's different)
  PHX::MDField<ScalarT> f(fieldName, dl->qp_scalar); field = f;
  PHX::MDField<ScalarT> fg(fieldGradientName, dl->qp_vector); fieldGradient = fg;

  if(!bReturnSameField) {
    PHX::MDField<ScalarT> fr(retFieldName, dl->qp_scalar); retField = fr; }


  //! add dependent fields
  this->addDependentField(field);
  this->addDependentField(fieldGradient);
  this->addDependentField(coordVec);
  this->addDependentField(coordVec_vertices);
  this->addDependentField(weights);
  if(!bReturnSameField) this->addDependentField(retField);

  std::string responseID = "QCAD Saddle Value";
  this->setName(responseID + PHX::TypeString<EvalT>::value);

  /*//! response evaluator must evaluate dummy operation
  Teuchos::RCP<PHX::DataLayout> dummy_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout");
  
  response_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(responseID, dummy_dl));
  this->addEvaluatedField(*response_operation);*/

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);

  int responseSize = 5;
  int worksetSize = dl->qp_scalar->dimension(0);
  Teuchos::RCP<PHX::DataLayout> global_response_layout =
    Teuchos::rcp(new PHX::MDALayout<Dim>(responseSize));
  Teuchos::RCP<PHX::DataLayout> local_response_layout =
    Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(worksetSize, responseSize));


  std::string local_response_name = 
    fieldName + " Local Response Saddle Value";
  std::string global_response_name = 
    fieldName + " Global Response Saddle Value";

  PHX::Tag<ScalarT> local_response_tag(local_response_name, 
				       local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, 
					global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(fieldGradient,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(coordVec_vertices,fm);
  this->utils.setFieldData(weights,fm);
  if(!bReturnSameField) this->utils.setFieldData(retField,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  const int MAX_DIMS = 3;
  ScalarT fieldVal, retFieldVal, cellVol, cellArea;
  std::vector<ScalarT> fieldGrad(numDims, 0.0);
  double dblAvgCoords[MAX_DIMS], dblFieldGrad[MAX_DIMS], dblFieldVal, dblCellArea, dblMaxZ;

  if(svResponseFn->getMode() == "Point location") {    
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {

      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);
      getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);
      dblFieldVal = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
      svResponseFn->addBeginPointData(workset.EBName, dblAvgCoords, dblFieldVal);
      svResponseFn->addEndPointData(workset.EBName, dblAvgCoords, dblFieldVal);
    }
  }
  else if(svResponseFn->getMode() == "Collect image point data") {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);

      if(svResponseFn->pointIsInImagePtRegion(dblAvgCoords, dblMaxZ)) {	
	getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);

	dblFieldVal = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
	for (std::size_t k=0; k < numDims; ++k)
	  dblFieldGrad[k] = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldGrad[k]);

	svResponseFn->addImagePointData(dblAvgCoords, dblFieldVal, dblFieldGrad);
      }
    }
  }
  else if(svResponseFn->getMode() == "Collect final image point data") {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);

      if(svResponseFn->pointIsInImagePtRegion(dblAvgCoords, dblMaxZ)) {	
	getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);

	dblFieldVal = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
	svResponseFn->addFinalImagePointData(dblAvgCoords, dblFieldVal);
      }
    }
  }
  else if(svResponseFn->getMode() == "Accumulate all field data") {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);

      if(svResponseFn->pointIsInAccumRegion(dblAvgCoords, dblMaxZ)) {	
	getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);

	dblFieldVal = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
	for (std::size_t k=0; k < numDims; ++k)
	  dblFieldGrad[k] = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldGrad[k]);
	svResponseFn->accumulatePointData(dblAvgCoords, dblFieldVal, dblFieldGrad);
      }
    }
  }
  else if(svResponseFn->getMode() == "Fill saddle point") {
    double wt;
    double totalWt = svResponseFn->getTotalSaddlePointWeight();

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);

      if(svResponseFn->pointIsInImagePtRegion(dblAvgCoords, dblMaxZ)) {	
	
	if( (wt = svResponseFn->getSaddlePointWeight(dblAvgCoords)) > 0.0) {
	  getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);
	  wt /= totalWt;

	  // Return field value
	  this->local_response(cell,0) += wt*retFieldVal;
	  this->global_response[0] += wt*retFieldVal;

	  // Field value (field searched for saddle point)
	  this->local_response(cell,1) += wt*fieldVal;
	  this->global_response[1] += wt*fieldVal;

	  this->global_response[2] = 0.0; // x-coord -- written later: would just be a MeshScalar anyway
	  this->global_response[3] = 0.0; // y-coord -- written later: would just be a MeshScalar anyway
	  this->global_response[4] = 0.0; // z-coord -- written later: would just be a MeshScalar anyway
	}
      }
    }
  }
  else if(svResponseFn->getMode() == "Level set data collection") {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      getAvgCellCoordinates(coordVec, cell, dblAvgCoords, dblMaxZ);

      if(svResponseFn->pointIsInLevelSetRegion(dblAvgCoords,dblMaxZ)) {	
	getCellQuantities(cell, cellVol, fieldVal, retFieldVal, fieldGrad);
	getCellArea(cell, cellArea);

	dblFieldVal = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
	dblCellArea  = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(cellVol);
	for (std::size_t k=0; k < numDims; ++k)
	  dblFieldGrad[k] = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldGrad[k]);
	svResponseFn->accumulateLevelSetData(dblAvgCoords, dblFieldVal, dblCellArea);
      }
    }
  }
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
		      << "Error!  Invalid mode: " << svResponseFn->getMode() << std::endl); 

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

//OLD: Keep for reference for now:
    // get a volume or area used for computing an average cell linear size
    /*if( bLateralVolumes ) {
      std::vector<ScalarT> maxCoord(3,-1e10);
      std::vector<ScalarT> minCoord(3,+1e10);

      for (std::size_t v=0; v < numVertices; ++v) {
	for (std::size_t k=0; k < numDims; ++k) {
	  if(maxCoord[k] < coordVec_vertices(cell,v,k)) maxCoord[k] = coordVec_vertices(cell,v,k);
	  if(minCoord[k] > coordVec_vertices(cell,v,k)) minCoord[k] = coordVec_vertices(cell,v,k);
	}
      }

      retCellVol = 1.0;
      for (std::size_t k=0; k < numDims && k < 2; ++k)  //limit to at most 2 dimensions
	retCellVol *= (maxCoord[k] - minCoord[k]);
    }
    else retCellVol = cellVol;
    */


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // only care about global response in "Fill saddle point" mode
  if(svResponseFn->getMode() == "Fill saddle point") {

    // Add contributions across processors
    Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
      workset.serializerManager.template getValue<EvalT>();

    // we cannot pass the same object for both the send and receive buffers in reduceAll call
    // creating a copy of the global_response, not a view
    std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
    PHX::MDField<ScalarT> partial_response(this->global_response);
    partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

    Teuchos::reduceAll(
      *workset.comm, *serializer, Teuchos::REDUCE_SUM,
      this->global_response.size(), &partial_response[0],
      &this->global_response[0]);

    // Copy in position of saddle point here (no derivative info yet)
    const double* pt = svResponseFn->getSaddlePointPosition();
    for(std::size_t i=0; i<numDims; i++) 
      this->global_response[2+i] = pt[i];

    if(retFieldName == "current" &&
       //(QCAD::EvaluatorTools<EvalT,Traits>::getEvalType() == "Tangent" ||
       // QCAD::EvaluatorTools<EvalT,Traits>::getEvalType() == "Residual"))
       (QCAD::EvaluatorTools<EvalT,Traits>::getEvalType() == "Residual") )  
    {
      // We only really need to evaluate the current when computing the final response values,
      // which is for the "Tangent" or "Residual" evaluation type (depeding on whether
      // sensitivities are being computed).  It would be nice to have a cleaner
      // way of implementing a response whose algorithm cannot support AD types. (EGN)
      
      this->global_response[1] = this->global_response[0];
      this->global_response[0] = svResponseFn->getCurrent(lattTemp, materialDB);
    }
	
    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
  }
}


// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseSaddleValue<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseSaddleValue Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Scalar field on which to find saddle point");
  validPL->set<std::string>("Field Gradient Name", "", "Gradient of field on which to find saddle point");
  validPL->set<std::string>("Return Field Name", "<field name>", "Scalar field to return value from");
  validPL->set<double>("Field Scaling Factor", 1.0, "Scaling factor for field on which to find saddle point");
  validPL->set<double>("Field Gradient Scaling Factor", 1.0, "Scaling factor for field gradient");
  validPL->set<double>("Return Field Scaling Factor", 1.0, "Scaling factor for return field");

  validPL->set<int>("Number of Image Points", 10, "Number of image points to use, including the two endpoints");
  validPL->set<double>("Image Point Size", 1.0, "Size of image points, modeled as gaussian weight distribution");
  validPL->set<int>("Maximum Iterations", 100, "Maximum number of NEB iterations");
  validPL->set<int>("Backtrace After Iteration", 10000000, "Backtrace, i.e., don't let grad(highest pt) increase, after this iteration");
  validPL->set<double>("Max Time Step", 1.0, "Maximum (and initial) time step");
  validPL->set<double>("Min Time Step", 0.002, "Minimum time step");
  validPL->set<double>("Convergence Tolerance", 1e-5, "Convergence criterion when |grad| of saddle is below this number");
  validPL->set<double>("Min Spring Constant", 1.0, "Minimum spring constant used between image points (initial time)");
  validPL->set<double>("Max Spring Constant", 1.0, "Maximum spring constant used between image points (final time)");
  validPL->set<std::string>("Output Filename", "", "Filename to receive elastic band points and values at given interval");
  validPL->set<int>("Output Interval", 0, "Output elastic band points every <output interval> iterations");
  validPL->set<std::string>("Debug Filename", "", "Filename for algorithm debug info");
  validPL->set<bool>("Append Output", false, "If true, output is appended to Output Filename (if it exists)");
  validPL->set<bool>("Climbing NEB", true, "Whether or not to use the climbing NEB algorithm");
  validPL->set<double>("Anti-Kink Factor", 0.0, "Factor between 0 and 1 giving about of perpendicular spring force to inclue");
  validPL->set<bool>("Aggregate Worksets", false, "Whether or not to store off a proc's worksets locally.  Increased speed but requires more memory");
  validPL->set<bool>("Adaptive Image Point Size", false, "Whether or not image point sizes should adapt to local mesh density");
  validPL->set<double>("Adaptive Min Point Weight", 0.5, "Minimum desirable point weight when adaptively choosing image point sizes");
  validPL->set<double>("Adaptive Max Point Weight", 5.0, "Maximum desirable point weight when adaptively choosing image point sizes");

  validPL->set<double>("Levelset Field Cutoff Factor", 1.0, "Fraction of field range to use as cutoff in level set algorithm");
  validPL->set<double>("Levelset Minimum Pool Depth Factor", 1.0, "Fraction of automatic value to use as minimum pool depth level set algorithm");
  validPL->set<double>("Levelset Distance Cutoff Factor", 1.0, "Fraction of avg cell length to use as cutoff in level set algorithm");
  validPL->set<double>("Levelset Radius", 0.0, "Radius around image point to use as level-set domain (zero == don't use level set");

  validPL->set<double>("z min", 0.0, "Domain minimum z coordinate");
  validPL->set<double>("z max", 0.0, "Domain maximum z coordinate");
  validPL->set<double>("Lock to z-coord", 0.0, "z-coordinate to lock elastic band to, making a 3D problem into 2D");

  validPL->set<int>("Maximum Number of Final Points", 0, "Maximum number of final points to use.  Zero indicates no final points are used and data is just returned at image points.");

  validPL->set<Teuchos::Array<double> >("Begin Point", Teuchos::Array<double>(), "Beginning point of elastic band");
  validPL->set<std::string>("Begin Element Block", "", "Element block name whose minimum marks the elastic band's beginning");
  validPL->sublist("Begin Polygon", false, "Beginning polygon sublist");

  validPL->set<Teuchos::Array<double> >("End Point", Teuchos::Array<double>(), "Ending point of elastic band");
  validPL->set<std::string>("End Element Block", "", "Element block name whose minimum marks the elastic band's ending");
  validPL->sublist("End Polygon", false, "Ending polygon sublist");

  validPL->set<double>("Percent to Shorten Begin", 0.0, "Percentage of total or half path (if guessed pt) to shorten the beginning of the path");
  validPL->set<double>("Percent to Shorten End", 0.0, "Percentage of total or half path (if guessed pt) to shorten the end of the path");

  validPL->set<Teuchos::Array<double> >("Saddle Point Guess", Teuchos::Array<double>(), "Estimate of where the saddle point lies");

  validPL->set<double>("GF-CBR Method Energy Cutoff Offset", 0, "Value [in eV] added to the maximum energy integrated over in Green's Function - Contact Block Reduction method for obtaining current to obtain the cutoff energy, which sets the largest eigenvalue needed in the tight binding diagonalization part of the method");
  validPL->set<double>("GF-CBR Method Grid Spacing", 0.0005, "Uniform 1D grid spacing for GF-CBR current calculation - given in mesh units");

  validPL->set<bool>("GF-CBR Method Vds Sweep", false, "Specify if want to sweep a range of Vds values or just want one Vds");
  validPL->set<double>("GF-CBR Method Vds Initial Value", 0., "Initial Vds value [V] when sweeping Vds is true");
  validPL->set<double>("GF-CBR Method Vds Final Value", 0., "Final Vds value [V]");
  validPL->set<int>("GF-CBR Method Vds Steps", 10, "Number of Vds steps going from initial to final values");
  validPL->set<std::string>("GF-CBR Method Eigensolver", "", "Eigensolver used by the GF-CBR method");
  
  validPL->set<int>("Debug Mode", 0, "Print verbose debug messages to stdout");
  validPL->set< Teuchos::RCP<QCAD::SaddleValueResponseFunction> >("Response Function", Teuchos::null, "Saddle value response function");

  validPL->set<std::string>("Description", "", "Description of this response used by post processors");
  
  return validPL;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
getCellQuantities(const std::size_t cell, typename EvalT::ScalarT& cellVol, typename EvalT::ScalarT& fieldVal, 
		    typename EvalT::ScalarT& retFieldVal, std::vector<typename EvalT::ScalarT>& fieldGrad) const
{
  cellVol = 0.0;
  fieldVal = 0.0;
  retFieldVal = 0.0;
  for (std::size_t k=0; k < numDims; ++k) fieldGrad[k] = 0.0;

  // Get the cell volume
  for (std::size_t qp=0; qp < numQPs; ++qp)
    cellVol += weights(cell,qp);

  // Get cell average value of field
  for (std::size_t qp=0; qp < numQPs; ++qp)
    fieldVal += field(cell,qp) * weights(cell,qp);
  fieldVal *= scaling / cellVol;  

  // Get the cell average of the field to return
  if(bReturnSameField) {
    retFieldVal = retScaling * fieldVal;
  }
  else {
    for (std::size_t qp=0; qp < numQPs; ++qp)
      retFieldVal += retField(cell,qp) * weights(cell,qp);
    retFieldVal *= retScaling / cellVol;
  }

  // Get cell average of the gradient field
  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t k=0; k < numDims; ++k) 
      fieldGrad[k] += fieldGradient(cell,qp,k) * weights(cell,qp);
  }
  for (std::size_t k=0; k < numDims; ++k) fieldGrad[k] *= gradScaling / cellVol; 

  return;
}

template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
getCellArea(const std::size_t cell, typename EvalT::ScalarT& cellArea) const
{
  std::vector<ScalarT> maxCoord(3,-1e10);
  std::vector<ScalarT> minCoord(3,+1e10);

  for (std::size_t v=0; v < numVertices; ++v) {
    for (std::size_t k=0; k < numDims; ++k) {
      if(maxCoord[k] < coordVec_vertices(cell,v,k)) maxCoord[k] = coordVec_vertices(cell,v,k);
      if(minCoord[k] > coordVec_vertices(cell,v,k)) minCoord[k] = coordVec_vertices(cell,v,k);
    }
  }

  cellArea = 1.0;
  for (std::size_t k=0; k < numDims && k < 2; ++k)  //limit to at most 2 dimensions
    cellArea *= (maxCoord[k] - minCoord[k]);
}



// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
  getAvgCellCoordinates(PHX::MDField<typename EvalT::MeshScalarT,Cell,QuadPoint,Dim> coordVec,
			const std::size_t cell, double* dblAvgCoords, double& dblMaxZ) const
{
  std::vector<MeshScalarT> avgCoord(numDims, 0.0); //just a double?
  

  //Get average cell coordinate (avg of qps)
  for (std::size_t k=0; k < numDims; ++k) avgCoord[k] = 0.0;
  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t k=0; k < numDims; ++k) 
      avgCoord[k] += coordVec(cell,qp,k);
  }
  for (std::size_t k=0; k < numDims; ++k) {
    avgCoord[k] /= numQPs;
    dblAvgCoords[k] = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(avgCoord[k]);
  }

  //Get maximium z-coordinate in cell
  if(numDims > 2) {
    MeshScalarT maxZ = -1e10;
    for (std::size_t v=0; v < numVertices; ++v) {
      if(maxZ < coordVec_vertices(cell,v,2)) maxZ = coordVec_vertices(cell,v,2);
    }
    dblMaxZ = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(maxZ);
  }
  else dblMaxZ = 0.0;  //Just set maximum Z-coord to zero if < 3 dimensions

}
