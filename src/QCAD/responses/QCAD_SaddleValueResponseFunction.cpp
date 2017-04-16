//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_Array.hpp>
#if defined(ALBANY_EPETRA)
#include <Epetra_LocalMap.h>
#endif
#include "Albany_Utils.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_DistObject.hpp"
#include "Tpetra_Map.hpp"
#include "QCAD_GreensFunctionTunneling.hpp"
#include <fstream>
#include "Petra_Converters.hpp" 

//! Helper function prototypes
namespace QCAD 
{
  //Moved to MathVector.hpp
  //bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const QCAD::mathVector& pt);
  //bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const double* pt);

#if defined(ALBANY_EPETRA)
  void gatherVector(std::vector<double>& v, std::vector<double>& gv,
		    const Epetra_Comm& comm);
#endif
  void gatherVectorT(std::vector<double>& v, std::vector<double>& gv,
		    Teuchos::RCP<const Teuchos::Comm<int> >& commT);
  void getOrdering(const std::vector<double>& v, std::vector<int>& ordering);
  bool lessOp(std::pair<std::size_t, double> const& a,
	      std::pair<std::size_t, double> const& b);
  double averageOfVector(const std::vector<double>& v);
  double distance(const std::vector<double>* vCoords, int ind1, int ind2, std::size_t nDims);
}

QCAD::SaddleValueResponseFunction::
SaddleValueResponseFunction(
  const Teuchos::RCP<Albany::Application>& application,
  const Teuchos::RCP<Albany::AbstractProblem>& problem,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
  const Teuchos::RCP<Albany::StateManager>& stateMgr,
  Teuchos::ParameterList& params) : 
  Albany::FieldManagerScalarResponseFunction(application, problem, ms, stateMgr),
  numDims(problem->spatialDimension())
{
  TEUCHOS_TEST_FOR_EXCEPTION (numDims < 2 || numDims > 3, Teuchos::Exceptions::InvalidParameter, std::endl 
	      << "Saddle Point not implemented for " << numDims << " dimensions." << std::endl); 

  params.set("Response Function", Teuchos::rcp(this,false));

  Teuchos::Array<double> ar;
  Teuchos::RCP<const Teuchos_Comm> commT = application->getComm(); 
#if defined(ALBANY_EPETRA)
  comm = application->getEpetraComm();
#endif
  imagePtSize   = params.get<double>("Image Point Size", 0.01);
  nImagePts     = params.get<int>("Number of Image Points", 10);
  maxTimeStep   = params.get<double>("Max Time Step", 1.0);
  minTimeStep   = params.get<double>("Min Time Step", 0.002);
  maxIterations = params.get<int>("Maximum Iterations", 100);
  backtraceAfterIters = params.get<int>("Backtrace After Iteration", 10000000);
  convergeTolerance   = params.get<double>("Convergence Tolerance", 1e-5);
  minSpringConstant   = params.get<double>("Min Spring Constant", 1.0);
  maxSpringConstant   = params.get<double>("Max Spring Constant", 1.0);
  outputFilename = params.get<std::string>("Output Filename", "");
  debugFilename  = params.get<std::string>("Debug Filename", "");
  appendOutput   = params.get<bool>("Append Output", false);
  nEvery         = params.get<int>("Output Interval", 0);
  bClimbing      = params.get<bool>("Climbing NEB", true);
  antiKinkFactor = params.get<double>("Anti-Kink Factor", 0.0);
  bAggregateWorksets = params.get<bool>("Aggregate Worksets", false);
  bAdaptivePointSize = params.get<bool>("Adaptive Image Point Size", false);
  minAdaptivePointWt = params.get<double>("Adaptive Min Point Weight", 5);
  maxAdaptivePointWt = params.get<double>("Adaptive Max Point Weight", 10);
  shortenBeginPc = params.get<double>("Percent to Shorten Begin", 0);
  shortenEndPc   = params.get<double>("Percent to Shorten End", 0);

  fieldCutoffFctr = params.get<double>("Levelset Field Cutoff Factor", 1.0);
  minPoolDepthFctr = params.get<double>("Levelset Minimum Pool Depth Factor", 1.0);
  distanceCutoffFctr = params.get<double>("Levelset Distance Cutoff Factor", 1.0);
  levelSetRadius = params.get<double>("Levelset Radius", 0);

  // set default maxFinalPts to nImagePts
  maxFinalPts = params.get<int>("Maximum Number of Final Points", nImagePts);
  gfGridSpacing  = params.get<double>("GF-CBR Method Grid Spacing", 0.0005);
  fieldScaling = params.get<double>("Field Scaling Factor", 1.0);
  
  // set Vds information
  bSweepVds = params.get<bool>("GF-CBR Method Vds Sweep", false);
  initVds = params.get<double>("GF-CBR Method Vds Initial Value", 0.0);
  finalVds = params.get<double>("GF-CBR Method Vds Final Value", 0.0);
  stepsVds = params.get<int>("GF-CBR Method Vds Steps", 0);

  // specify the eigensolver to be used for the GF-CBR calculation
  gfEigensolver = params.get<std::string>("GF-CBR Method Eigensolver", "tql2");
  std::cout << "gfEigensolver = " << gfEigensolver << std::endl; 
  
  bGetCurrent = (params.get<std::string>("Return Field Name", "") == "current");
  
  // set default value to 0.5 eV (always want a positive value)
  current_Ecutoff_offset_from_Emax = params.get<double>("GF-CBR Method Energy Cutoff Offset", 0.5);

  if(backtraceAfterIters < 0) backtraceAfterIters = 10000000;
  else if(backtraceAfterIters <= 1) backtraceAfterIters = 2; // can't backtrace until the second iteration

  bLockToPlane = false;
  if(params.isParameter("Lock to z-coord")) {
    bLockToPlane = true;
    lockedZ = params.get<double>("Lock to z-coord");
  }

  iSaddlePt = -1;        //clear "found" saddle point index
  returnFieldVal = -1.0; //init to nonzero is important - so doesn't "match" default init

  //Beginning target region
  if(params.isParameter("Begin Point")) {
    beginRegionType = "Point";
    ar = params.get<Teuchos::Array<double> >("Begin Point");
    TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)numDims, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Begin Point does not have " << numDims << " elements" << std::endl); 
    beginPolygon.resize(1); beginPolygon[0].resize(numDims);
    for(std::size_t i=0; i<numDims; i++) beginPolygon[0][i] = ar[i];
  }
  else if(params.isParameter("Begin Element Block")) {
    beginRegionType = "Element Block";
    beginElementBlock = params.get<std::string>("Begin Element Block");
  }
  else if(params.isSublist("Begin Polygon")) {
    beginRegionType = "Polygon";

    Teuchos::ParameterList& polyList = params.sublist("Begin Polygon");
    int nPts = polyList.get<int>("Number of Points");
    beginPolygon.resize(nPts); 

    for(int i=0; i<nPts; i++) {
      beginPolygon[i].resize(numDims);
      ar = polyList.get<Teuchos::Array<double> >( Albany::strint("Point",i) );
      TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)numDims, Teuchos::Exceptions::InvalidParameter, std::endl 
				  << "Begin polygon point does not have " << numDims << " elements" << std::endl); 
      for(std::size_t k=0; k<numDims; k++) beginPolygon[i][k] = ar[k];
    }
  }
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				  << "No beginning region specified for saddle pt" << std::endl); 

  

  //Ending target region
  if(params.isParameter("End Point")) {
    endRegionType = "Point";
    ar = params.get<Teuchos::Array<double> >("End Point");
    TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)numDims, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "End Point does not have " << numDims << " elements" << std::endl); 
    endPolygon.resize(1); endPolygon[0].resize(numDims);
    for(std::size_t i=0; i<numDims; i++) endPolygon[0][i] = ar[i];
  }
  else if(params.isParameter("End Element Block")) {
    endRegionType = "Element Block";
    endElementBlock = params.get<std::string>("End Element Block");
  }
  else if(params.isSublist("End Polygon")) {
    endRegionType = "Polygon";
    
    Teuchos::ParameterList& polyList = params.sublist("End Polygon");
    int nPts = polyList.get<int>("Number of Points");
    endPolygon.resize(nPts); 
    
    for(int i=0; i<nPts; i++) {
      endPolygon[i].resize(numDims);
      ar = polyList.get<Teuchos::Array<double> >( Albany::strint("Point",i) );
      TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)numDims, Teuchos::Exceptions::InvalidParameter, std::endl 
				  << "End polygon point does not have " << numDims << " elements" << std::endl); 
      for(std::size_t k=0; k<numDims; k++) endPolygon[i][k] = ar[k];
    }
  }
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				  << "No ending region specified for saddle pt" << std::endl); 
  

  //Guess at the saddle point
  saddleGuessGiven = false;
  if(params.isParameter("Saddle Point Guess")) {
    saddleGuessGiven = true;
    ar = params.get<Teuchos::Array<double> >("Saddle Point Guess");
    TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)numDims, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Saddle point guess does not have " << numDims << " elements" << std::endl); 
    saddlePointGuess.resize(numDims);
    for(std::size_t i=0; i<numDims; i++) saddlePointGuess[i] = ar[i];
  }

  debugMode = params.get<int>("Debug Mode",0);

  imagePts.resize(nImagePts);
  imagePtValues.resize(nImagePts);
  imagePtWeights.resize(nImagePts);
  imagePtGradComps.resize(nImagePts*numDims);

  // Add allowed z-range if in 3D (lateral volume assumed)
  //  - rest (xmin, etc) computed dynamically
  if(numDims > 2) {
    zmin = params.get<double>("z min");
    zmax = params.get<double>("z max");
  }  

  this->setup(params);
  this->num_responses = 5;
}

QCAD::SaddleValueResponseFunction::
~SaddleValueResponseFunction()
{
}

unsigned int
QCAD::SaddleValueResponseFunction::
numResponses() const 
{
  return this->num_responses;  // returnFieldValue, fieldValue, saddleX, saddleY, saddleZ
}

void
QCAD::SaddleValueResponseFunction::
evaluateResponseT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector* xdotdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     Tpetra_Vector& gT)
{
  Teuchos::RCP<const Teuchos::Comm<int> > commT = xT.getMap()->getComm();

  int dbMode = (commT->getRank() == 0) ? debugMode : 0;
  if(commT->getRank() != 0) outputFilename = ""; //Only root process outputs to files
  if(commT->getRank() != 0) debugFilename = ""; //Only root process outputs to files
  
  TEUCHOS_TEST_FOR_EXCEPTION (nImagePts < 2, Teuchos::Exceptions::InvalidParameter, std::endl 
	      << "Saddle Point needs more than 2 image pts (" << nImagePts << " given)" << std::endl); 

  // Find saddle point in stages:
 
  //  1) Initialize image points
  initializeImagePointsT(current_time, xdotT, xT, p, gT, dbMode);
  
  if(maxIterations > 0) {
    //  2) Perform Nudged Elastic Band (NEB) algorithm on image points (iterative)
    doNudgedElasticBandT(current_time, xdotT, xT, p, gT, dbMode);
  }
  else {
    // If no NEB iteractions, choose center image point as saddle point
    int nFirstLeg = (nImagePts+1)/2, iCenter = nFirstLeg-1;
    iSaddlePt = iCenter; //don't need to check for positive weight at this point
  }

//  3) Perform level-set method in a radius around saddle image point
doLevelSetT(current_time, xdotT, xT, p, gT, dbMode);
/*
//  4) Fill response (g-vector) with values near the highest image point
fillSaddlePointData(current_time, xdot, x, p, g, dbMode);


*/

  return;
}


#if defined(ALBANY_EPETRA)
void
QCAD::SaddleValueResponseFunction::
initializeImagePoints(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g, int dbMode)
{
  // 1) Determine initial and final points depending on region type
  //     - Point: take point given directly
  //     - Element Block: take minimum point within the specified element block (and allowed z-range)
  //     - Polygon: take minimum point within specified 2D polygon and allowed z-range
  
  if(dbMode > 1) std::cout << "Saddle Point:  Beginning end point location" << std::endl;

    // Initialize intial/final points
  imagePts[0].init(numDims, imagePtSize);
  imagePts[nImagePts-1].init(numDims, imagePtSize);

  mode = "Point location";
  
  Teuchos::RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);

  //convert xdot_poisson and x_poisson to Tpetra  
  Teuchos::RCP<const Tpetra_Vector> xT, xdotT; 
  xT  = Petra::EpetraVector_To_TpetraVectorConst(x, commT); 
  if (xdot != NULL)  
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  Teuchos::RCP<Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorNonConst(g, commT); 

  Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
	current_time, xdotT.get(), NULL, *xT, p, *gT);
  if(dbMode > 2) std::cout << "Saddle Point:   -- done evaluation" << std::endl;

  if(beginRegionType == "Point") {
    imagePts[0].coords = beginPolygon[0];
  }
  else { 

    //MPI: get global min for begin point
    double globalMin; int procToBcast, winner;
    comm->MinAll( &imagePts[0].value, &globalMin, 1);
    if( fabs(imagePts[0].value - globalMin) < 1e-8 ) 
      procToBcast = comm->MyPID();
    else procToBcast = -1;

    comm->MaxAll( &procToBcast, &winner, 1 );
    comm->Broadcast( imagePts[0].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[0].value = globalMin;                               //no need to broadcast winner's value
  }

  if(endRegionType   == "Point") {
    imagePts[nImagePts-1].coords = endPolygon[0];
  }
  else { 

    //MPI: get global min for end point
    double globalMin; int procToBcast, winner;
    comm->MinAll( &imagePts[nImagePts-1].value, &globalMin, 1);
    if( fabs(imagePts[nImagePts-1].value - globalMin) < 1e-8 ) 
      procToBcast = comm->MyPID();
    else procToBcast = -1;

    comm->MaxAll( &procToBcast, &winner, 1 );
    comm->Broadcast( imagePts[nImagePts-1].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[nImagePts-1].value = globalMin;                               //no need to broadcast winner's value
  }

  //! Shorten beginning and end of path if requested (used to move begin/end point off of a contact region in QCAD)
  if(shortenBeginPc > 1e-6) {
     if(saddleGuessGiven)
       imagePts[0].coords = imagePts[0].coords + (saddlePointGuess - imagePts[0].coords) * (shortenBeginPc/100.0);
     else
       imagePts[0].coords = imagePts[0].coords + (imagePts[nImagePts-1].coords - imagePts[0].coords) * (shortenBeginPc/100.0);
  }
  if(shortenEndPc > 1e-6) {
     if(saddleGuessGiven)
       imagePts[nImagePts-1].coords = imagePts[nImagePts-1].coords + (saddlePointGuess - imagePts[nImagePts-1].coords) * (shortenEndPc/100.0);
     else
       imagePts[nImagePts-1].coords = imagePts[nImagePts-1].coords + (imagePts[0].coords - imagePts[nImagePts-1].coords) * (shortenEndPc/100.0);
  }

  if(dbMode > 2) std::cout << "Saddle Point:   -- done begin/end point initialization" << std::endl;

  //! Initialize Image Points:  
  //   interpolate between initial and final points (and possibly guess point) 
  //   to get all the image points
  const mathVector& initialPt = imagePts[0].coords;
  const mathVector& finalPt   = imagePts[nImagePts-1].coords;

  // Lock z-coordinate of initial and final points (and therefore of the rest of the points) if requested
  if(bLockToPlane && numDims > 2)
    imagePts[0].coords[2] = imagePts[nImagePts-1].coords[2] = lockedZ;

  if(saddleGuessGiven) {

    // two line segements (legs) initialPt -> guess, guess -> finalPt
    int nFirstLeg = (nImagePts+1)/2, nSecondLeg = nImagePts - nFirstLeg + 1; // +1 because both legs include middle pt
    for(int i=1; i<nFirstLeg-1; i++) {
      double s = i * 1.0/(nFirstLeg-1);
      imagePts[i].init(initialPt + (saddlePointGuess - initialPt) * s, imagePtSize);
    }
    for(int i=0; i<nSecondLeg-1; i++) {
      double s = i * 1.0/(nSecondLeg-1);
      imagePts[i+nFirstLeg-1].init(saddlePointGuess + (finalPt - saddlePointGuess) * s, imagePtSize);
    }
  }
  else {

    // one line segment initialPt -> finalPt
    for(std::size_t i=1; i<nImagePts-1; i++) {
      double s = i * 1.0/(nImagePts-1);   // nIntervals = nImagePts-1
      imagePts[i].init(initialPt + (finalPt - initialPt) * s, imagePtSize);
    }     
  }
 
  // Print initial point locations to stdout if requested
  if(dbMode > 1) {
    for(std::size_t i=0; i<nImagePts; i++)
      std::cout << "Saddle Point:   -- imagePt[" << i << "] = " << imagePts[i].coords << std::endl;
  }

  // If we aggregate workset data then call evaluator once more to accumulate 
  //  field and coordinate data into vFieldValues and vCoords members.
  if(bAggregateWorksets) {
    vFieldValues.clear();
    vCoords.clear();
    vGrads.clear();

    mode = "Accumulate all field data";
    Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				    current_time, xdotT.get(), NULL, *xT, p, *gT);
    //No MPI here - each proc only holds all of it's worksets -- not other procs worksets
  }


  return;
}
#endif

void
QCAD::SaddleValueResponseFunction::
initializeImagePointsT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     Tpetra_Vector& gT, int dbMode)
{
  // 1) Determine initial and final points depending on region type
  //     - Point: take point given directly
  //     - Element Block: take minimum point within the specified element block (and allowed z-range)
  //     - Polygon: take minimum point within specified 2D polygon and allowed z-range
  
  Teuchos::RCP<const Teuchos::Comm<int> > commT = xT.getMap()->getComm();
  if(dbMode > 1) std::cout << "Saddle Point:  Beginning end point location" << std::endl;

    // Initialize intial/final points
  imagePts[0].init(numDims, imagePtSize);
  imagePts[nImagePts-1].init(numDims, imagePtSize);

  mode = "Point location";
  Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
	current_time, xdotT, NULL, xT, p, gT);
  if(dbMode > 2) std::cout << "Saddle Point:   -- done evaluation" << std::endl;

  if(beginRegionType == "Point") {
    imagePts[0].coords = beginPolygon[0];
  }
  else { 

    //MPI: get global min for begin point
    double globalMin; int procToBcast, winner;
    //comm.MinAll( &imagePts[0].value, &globalMin, 1);
    double imagePtsValue = imagePts[0].value; 
    Teuchos::reduceAll(*commT, Teuchos::REDUCE_MIN, imagePtsValue, Teuchos::ptr(&globalMin));
    if( fabs(imagePts[0].value - globalMin) < 1e-8 ) 
      procToBcast = commT->getRank();
    else procToBcast = -1;

    //comm.MaxAll( &procToBcast, &winner, 1 );
    int procToBcastInt = procToBcast; 
    Teuchos::reduceAll(*commT, Teuchos::REDUCE_MAX, procToBcastInt, Teuchos::ptr(&winner));
    double *imagePtsData = imagePts[0].coords.data();
    Teuchos::broadcast<LO, ST>(*commT, winner, imagePtsData);  
    //comm.Broadcast( imagePts[0].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[0].value = globalMin;                               //no need to broadcast winner's value
  }

  if(endRegionType   == "Point") {
    imagePts[nImagePts-1].coords = endPolygon[0];
  }
  else { 

    //MPI: get global min for end point
    double globalMin; int procToBcast, winner;
    //comm.MinAll( &imagePts[nImagePts-1].value, &globalMin, 1);
    double imagePtsValue = imagePts[nImagePts-1].value; 
    Teuchos::reduceAll(*commT, Teuchos::REDUCE_MIN, imagePtsValue, Teuchos::ptr(&globalMin));
    if( fabs(imagePts[nImagePts-1].value - globalMin) < 1e-8 ) { 
      //procToBcast = comm.MyPID();
      procToBcast = commT->getRank();
    }
    else procToBcast = -1;

    //comm.MaxAll( &procToBcast, &winner, 1 );
    int procToBCastInt = procToBcast; 
    Teuchos::reduceAll(*commT, Teuchos::REDUCE_MAX, procToBCastInt, Teuchos::ptr(&winner));
    double *imagePtsData = imagePts[nImagePts-1].coords.data(); 
    Teuchos::broadcast<LO, ST>(*commT, winner, imagePtsData); 
    //comm.Broadcast( imagePts[nImagePts-1].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[nImagePts-1].value = globalMin;                               //no need to broadcast winner's value
  }

  //! Shorten beginning and end of path if requested (used to move begin/end point off of a contact region in QCAD)
  if(shortenBeginPc > 1e-6) {
     if(saddleGuessGiven)
       imagePts[0].coords = imagePts[0].coords + (saddlePointGuess - imagePts[0].coords) * (shortenBeginPc/100.0);
     else
       imagePts[0].coords = imagePts[0].coords + (imagePts[nImagePts-1].coords - imagePts[0].coords) * (shortenBeginPc/100.0);
  }
  if(shortenEndPc > 1e-6) {
     if(saddleGuessGiven)
       imagePts[nImagePts-1].coords = imagePts[nImagePts-1].coords + (saddlePointGuess - imagePts[nImagePts-1].coords) * (shortenEndPc/100.0);
     else
       imagePts[nImagePts-1].coords = imagePts[nImagePts-1].coords + (imagePts[0].coords - imagePts[nImagePts-1].coords) * (shortenEndPc/100.0);
  }

  if(dbMode > 2) std::cout << "Saddle Point:   -- done begin/end point initialization" << std::endl;

  //! Initialize Image Points:  
  //   interpolate between initial and final points (and possibly guess point) 
  //   to get all the image points
  const mathVector& initialPt = imagePts[0].coords;
  const mathVector& finalPt   = imagePts[nImagePts-1].coords;

  // Lock z-coordinate of initial and final points (and therefore of the rest of the points) if requested
  if(bLockToPlane && numDims > 2)
    imagePts[0].coords[2] = imagePts[nImagePts-1].coords[2] = lockedZ;

  if(saddleGuessGiven) {

    // two line segements (legs) initialPt -> guess, guess -> finalPt
    int nFirstLeg = (nImagePts+1)/2, nSecondLeg = nImagePts - nFirstLeg + 1; // +1 because both legs include middle pt
    for(int i=1; i<nFirstLeg-1; i++) {
      double s = i * 1.0/(nFirstLeg-1);
      imagePts[i].init(initialPt + (saddlePointGuess - initialPt) * s, imagePtSize);
    }
    for(int i=0; i<nSecondLeg-1; i++) {
      double s = i * 1.0/(nSecondLeg-1);
      imagePts[i+nFirstLeg-1].init(saddlePointGuess + (finalPt - saddlePointGuess) * s, imagePtSize);
    }
  }
  else {

    // one line segment initialPt -> finalPt
    for(std::size_t i=1; i<nImagePts-1; i++) {
      double s = i * 1.0/(nImagePts-1);   // nIntervals = nImagePts-1
      imagePts[i].init(initialPt + (finalPt - initialPt) * s, imagePtSize);
    }     
  }
 
  // Print initial point locations to stdout if requested
  if(dbMode > 1) {
    for(std::size_t i=0; i<nImagePts; i++)
      std::cout << "Saddle Point:   -- imagePt[" << i << "] = " << imagePts[i].coords << std::endl;
  }

  // If we aggregate workset data then call evaluator once more to accumulate 
  //  field and coordinate data into vFieldValues and vCoords members.
  if(bAggregateWorksets) {
    vFieldValues.clear();
    vCoords.clear();
    vGrads.clear();

    mode = "Accumulate all field data";
    Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				    current_time, xdotT, NULL, xT, p, gT);
    //No MPI here - each proc only holds all of it's worksets -- not other procs worksets
  }
}

#if defined(ALBANY_EPETRA)
void
QCAD::SaddleValueResponseFunction::
initializeFinalImagePoints(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g, int dbMode)
{
  // Determine the locations of the "final" image points, which interpolate between the image points used
  //    in the nudged elastic band algorithm, and are used only as a means of getting more dense output data (more points along saddle path)
  
  if(dbMode > 1) std::cout << "Saddle Point:  Initializing Final Image Points" << std::endl;

  int maxPoints = maxFinalPts;    // maximum number of total final image points
  
  double* segmentLength = new double[nImagePts-1]; // segmentLength[i] == distance between imagePt[i] and imagePt[i+1]
  double lengthBefore = 0.0, lengthAfter = 0.0;    // path length before and after saddle point
  double radius = 0.0;
  int nPtsBefore = 0, nPtsAfter = 0, nFinalPts;

  // Get the distances along each leg of the current (final) saddle path
  for(std::size_t i = 0; i < nImagePts-1; i++) {
    segmentLength[i] = imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    radius += imagePts[i].radius;
    if( (int)i < iSaddlePt ) lengthBefore += segmentLength[i];
    else lengthAfter += segmentLength[i];
  }
  
  radius += imagePts[nImagePts-1].radius;
  radius /= nImagePts;  // average radius

/*
  // We'd like to put equal number of final points on each side of the saddle point.  Compute here how 
  //  many final points (fixed spacing) will lie on each side of the saddle point.
  if(maxPoints * ptSpacing < lengthBefore + lengthAfter) {
    if( maxPoints * ptSpacing / 2 > lengthBefore)
      nPtsBefore = int(lengthBefore / ptSpacing);
    else if( maxPoints * ptSpacing / 2 > lengthAfter)
      nPtsBefore = maxPoints - int(lengthAfter / ptSpacing);
    else nPtsBefore = maxPoints / 2;

    nPtsAfter = maxPoints - nPtsBefore;
  }
  else {
    nPtsBefore = int(lengthBefore / ptSpacing);
    nPtsAfter  = int(lengthAfter  / ptSpacing);
  }
*/

  // calculate point spacing for the entire saddle path and given maxPoints
  double ptSpacing = (lengthBefore + lengthAfter)/(maxPoints-1); 
  
  // nPtsBefore = int(lengthBefore / ptSpacing);
  // nPtsAfter  = int(lengthAfter  / ptSpacing);
  // nFinalPts = nPtsBefore + nPtsAfter + 1;     // one extra for "on/at" saddle point
  
  nFinalPts = maxPoints; 
  
  finalPts.resize(nFinalPts);
  finalPtValues.resize(nFinalPts);
  finalPtWeights.resize(nFinalPts);

  //! Initialize Final Image Points: interpolate between current image points
  //! If starting from the saddle point, the resulting finalPts.value vs pathLength 
  //! is shifted away from the imagePts.value vs pathLength curve.
  //! Hence, start from the first image point, and equally divide the entire saddle 
  //! into (maxPoints-1) pieces.
  
  // Assign the starting and ending image points to finalPts
  finalPts[0].init(imagePts[0].coords, imagePts[0].radius);
  finalPts[nFinalPts-1].init(imagePts[nImagePts-1].coords, imagePts[nImagePts-1].radius);
  
  double offset = ptSpacing;  
  int iCurFinalPt = 1;
  
  for(std::size_t i = 0; i < nImagePts-1; i++) {
    const mathVector& initialPt = imagePts[i].coords;
    const mathVector& v = (imagePts[i+1].coords - imagePts[i].coords) * (1.0/segmentLength[i]);  // normalized vector from initial -> final pt

	  if(segmentLength[i] > offset) {
	    int nPtSegs = int((segmentLength[i]-offset) / ptSpacing);
	    int nPts = nPtSegs + 1;
  	  double leftover = (segmentLength[i]-offset) - ptSpacing * nPtSegs;

	    for(int j = 0; j < nPts && iCurFinalPt < nFinalPts; j++) {
    	  finalPts[iCurFinalPt].init(initialPt + v * (ptSpacing * j + offset), radius );
      	iCurFinalPt++;
    	}
    	offset = ptSpacing - leftover; //how much to advance the first point of the next segment
    }
    else {
    	offset -= segmentLength[i];
    }
  }
  
  //If there are any leftover points, initialize them too
  for(int j = iCurFinalPt; j < nFinalPts; j++) 
    finalPts[j].init(imagePts[nImagePts-1].coords, imagePts[nImagePts-1].radius);
  

/*
  double offset = ptSpacing;
  int iCurFinalPt = nPtsBefore-1;
  for(int i = iSaddlePt-1; i >= 0; i--) {
    const mathVector& initialPt = imagePts[i+1].coords;
    const mathVector& v = (imagePts[i].coords - imagePts[i+1].coords) * (1.0/segmentLength[i]);  // normalized vector from initial -> final pt

    if(segmentLength[i] > offset) {
	    int nPtSegs = int((segmentLength[i]-offset) / ptSpacing);
  	  int nPts = nPtSegs + 1;
    	double leftover = (segmentLength[i]-offset) - ptSpacing * nPtSegs;

    	for(int j=0; j<nPts && iCurFinalPt >= 0; j++) {
      	//radius = (imagePts[i].radius + imagePts[i+1].radius)/2; // use average radius
      	finalPts[iCurFinalPt].init(initialPt + v * (ptSpacing * j + offset), radius );
      	iCurFinalPt--;
    	}
    	offset = ptSpacing - leftover; //how much to advance the first point of the next segment
    }
    else {
    	offset -= segmentLength[i];
    }
  }

  //If there are any leftover points (at beginning), initialize them too
  for(int j=0; j<=iCurFinalPt; j++) 
    finalPts[j].init(imagePts[0].coords, imagePts[0].radius);


  offset = ptSpacing;  //start initial point *after* saddle point this time
  iCurFinalPt = nPtsBefore+1;
  for(std::size_t i = iSaddlePt; i < nImagePts-1; i++) {
    const mathVector& initialPt = imagePts[i].coords;
    const mathVector& v = (imagePts[i+1].coords - imagePts[i].coords) * (1.0/segmentLength[i]);  // normalized vector from initial -> final pt

	  if(segmentLength[i] > offset) {
	    int nPtSegs = int((segmentLength[i]-offset) / ptSpacing);
	    int nPts = nPtSegs + 1;
  	  double leftover = (segmentLength[i]-offset) - ptSpacing * nPtSegs;

	    for(int j=0; j<nPts && iCurFinalPt < nFinalPts; j++) {
  	    // radius = (imagePts[i].radius + imagePts[i+1].radius)/2; // use average radius
    	  finalPts[iCurFinalPt].init(initialPt + v * (ptSpacing * j + offset), radius );
      	iCurFinalPt++;
    	}
    	offset = ptSpacing - leftover; //how much to advance the first point of the next segment
    }
    else {
    	offset -= segmentLength[i];
    }
  }

  //If there are any leftover points, initialize them too
  for(int j=iCurFinalPt; j<nFinalPts; j++) 
    finalPts[j].init(imagePts[nImagePts-1].coords, imagePts[nImagePts-1].radius);

*/

  return;
}

void
QCAD::SaddleValueResponseFunction::
doNudgedElasticBand(const double current_time,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    Epetra_Vector& g, int dbMode)
{
  //  2) Perform Nudged Elastic Band Algorithm to find saddle point.
  //      Iterate over field manager fills of each image point's value and gradient
  //       then update image point positions user Verlet algorithm

  std::size_t nIters, nInitialIterations;
  double dp, s;
  double gradScale, springScale, springBase;
  double avgForce=0, avgOpposingForce=0;
  double dt = maxTimeStep;
  double dt2 = dt*dt;
  double acceptedHighestPtGradNorm = -1.0, highestPtGradNorm;
  int iHighestPt, nConsecLowForceDiff=0;  

  mathVector tangent(numDims);  
  std::vector<mathVector> force(nImagePts), lastForce(nImagePts), lastPositions(nImagePts), lastVelocities(nImagePts);
  std::vector<double> springConstants(nImagePts-1, minSpringConstant);

  //initialize force variables and last positions
  for(std::size_t i=0; i<nImagePts; i++) {
    force[i].resize(numDims); force[i].fill(0.0);
    lastForce[i] = force[i];
    lastPositions[i] = imagePts[i].coords;
    lastVelocities[i] = imagePts[i].velocity;
  }

  //get distance between initial and final points
  double max_dCoords = imagePts[0].coords.distanceTo( imagePts[nImagePts-1].coords ) / nImagePts;

  nIters = 0;
  nInitialIterations = 20; // TODO: make into parameter?
  
  //Storage for aggrecated image point data (needed for MPI)
  double*  globalPtValues   = new double [nImagePts];
  double*  globalPtWeights  = new double [nImagePts];
  double*  globalPtGrads    = new double [nImagePts*numDims];

  //Write header to debug file
  std::fstream fDebug;
  if(debugFilename.length() > 0) {
    fDebug.open(debugFilename.c_str(), std::fstream::out);
    fDebug << "% HighestValue  HighestIndex  AverageForce  TimeStep"
	   << "  HighestPtGradNorm  AverageOpposingForce  SpringBase" << std::endl;
  }

  // Begin NEB iteration loop
  while( ++nIters <= maxIterations) {

    if(dbMode > 1) std::cout << "Saddle Point:  NEB Algorithm iteration " << nIters << " -----------------------" << std::endl;
    writeOutput(nIters);


    if(nIters > 1) {
      //Update coordinates and velocity using (modified) Verlet integration. Reset
      // the velocity to zero if it is directed opposite to force (reduces overshoot)
      for(std::size_t i=1; i<nImagePts-1; i++) {
	dp = imagePts[i].velocity.dot(force[i]);
	if(dp < 0) imagePts[i].velocity.fill(0.0);

	//save last position & velocity in case the new position brings 
	//  us outside the mesh or we need to backtrace
	lastPositions[i] = imagePts[i].coords; 
	lastVelocities[i] = imagePts[i].velocity; //Note: will be zero if force opposed velocity (above)

	// ** Update **
	mathVector dCoords = lastVelocities[i] * dt + force[i] * dt2 * 0.5;
	imagePts[i].coords = lastPositions[i] + dCoords;
	imagePts[i].velocity = lastVelocities[i] + force[i] * dt;
      }
    }

    getImagePointValues(current_time, xdot, x, p, g, 
			globalPtValues, globalPtWeights, globalPtGrads,
			lastPositions, dbMode);
    iHighestPt = getHighestPtIndex();
    highestPtGradNorm = imagePts[iHighestPt].grad.norm();

    // Setup scaling factors on first iteration
    if(nIters == 1) initialIterationSetup(gradScale, springScale, dbMode);

    // If in "backtrace" mode, require grad norm to decrease (or take minimum timestep)
    if(nIters > backtraceAfterIters) {
      while(dt > minTimeStep && highestPtGradNorm > acceptedHighestPtGradNorm) {

	//reduce dt
	dt = (dt/2 < minTimeStep) ? minTimeStep : dt/2;
	dt /= 2; dt2=dt*dt; 
	
	if(dbMode > 2) std::cout << "Saddle Point:  ** Backtrace dt => " << dt << std::endl;

	//Update coordinates and velocity using (modified) Verlet integration. Reset
	// the velocity to zero if it is directed opposite to force (reduces overshoot)
	for(std::size_t i=1; i<nImagePts-1; i++) {
	  mathVector dCoords = lastVelocities[i] * dt + force[i] * dt2 * 0.5;
	  imagePts[i].coords = lastPositions[i] + dCoords;
	  imagePts[i].velocity = lastVelocities[i] + force[i] * dt;
	}
	
	getImagePointValues(current_time, xdot, x, p, g, 
			    globalPtValues, globalPtWeights, globalPtGrads,
			    lastPositions, dbMode);
	iHighestPt = getHighestPtIndex();
	highestPtGradNorm = imagePts[iHighestPt].grad.norm();
      }

      if(dt == minTimeStep && highestPtGradNorm > acceptedHighestPtGradNorm && dbMode > 2)
	std::cout << "Saddle Point:  ** Warning: backtrace hit min dt == " << dt << std::endl;
    }	
    acceptedHighestPtGradNorm = highestPtGradNorm;

    // Compute spring base constant for this iteration
    s = ((double)nIters-1.0)/maxIterations;    
    springBase = springScale * ( (1.0-s)*minSpringConstant + s*maxSpringConstant ); 
    for(std::size_t i=0; i<nImagePts-1; i++) springConstants[i] = springBase;
	  
    avgForce = avgOpposingForce = 0.0;

    // Compute force acting on each image point
    for(std::size_t i=1; i<nImagePts-1; i++) {
      if(dbMode > 2) std::cout << std::endl << "Saddle Point:  >> Updating pt[" << i << "]:" << imagePts[i];

      // compute the tangent vector for the ith image point
      computeTangent(i, tangent, dbMode);

      // compute the force vector for the ith image point
      if((int)i == iHighestPt && bClimbing && nIters > nInitialIterations)
	computeClimbingForce(i, tangent, gradScale, force[i], dbMode);
      else
	computeForce(i, tangent, springConstants, gradScale, springScale,
		     force[i], dt, dt2, dbMode);

      // update avgForce and avgOpposingForce
      avgForce += force[i].norm();
      // Handle the 0 case so we don't divide by 0.
      const double lastForce_norm = lastForce[i].norm();
      dp = lastForce_norm == 0 ? 0 :
        force[i].dot(lastForce[i]) / (force[i].norm() * lastForce_norm);
      if( dp < 0 ) {  //if current force and last force point in "opposite" directions
	mathVector v = force[i] - lastForce[i];
	avgOpposingForce += v.norm() / (force[i].norm() + lastForce[i].norm());
	//avgOpposingForce += dp;  //an alternate implementation
      } 
    } // end of loop over image points 

    avgForce /= (nImagePts-2);
    avgOpposingForce /= (nImagePts-2);


    //print debug output
    if(dbMode > 1) 
      std::cout << "Saddle Point:  ** Summary:"
		<< "  highest val[" << iHighestPt << "] = " << imagePts[iHighestPt].value
		<< "  AverageForce = " << avgForce << "  dt = " << dt 
		<< "  gradNorm = " << imagePts[iHighestPt].grad.norm() 
		<< "  AvgOpposingForce = " << avgOpposingForce 
		<< "  SpringBase = " << springBase << std::endl;
    if(debugFilename.length() > 0)
      fDebug << imagePts[iHighestPt].value << "  " << iHighestPt << "  "
	     << avgForce << "  "  << dt << "  " << imagePts[iHighestPt].grad.norm() << "  "
	     << avgOpposingForce << "  " << springBase << std::endl;


    // Check for convergence in gradient norm
    if(imagePts[iHighestPt].grad.norm() < convergeTolerance) {
      if(dbMode > 2) std::cout << "Saddle Point:  ** Converged (grad norm " << 
	       imagePts[iHighestPt].grad.norm() << " < " << convergeTolerance << ")" << std::endl;
      break; // exit iterations loop
    }
    else if(nIters == maxIterations) break; //max iterations reached -- exit iterations loop now
                                            // (important so coords & radii don't get updated)

    // Save last force for next iteration
    for(std::size_t i=1; i<nImagePts-1; i++) lastForce[i] = force[i];
    
    // If all forces have remained in the same direction, tally this.  If this happens too many times
    //  increase dt, as this is a sign the time step is too small.
    if(avgOpposingForce < 1e-6) nConsecLowForceDiff++; else nConsecLowForceDiff = 0;
    if(nConsecLowForceDiff >= 3 && dt < maxTimeStep) { 
      dt *= 2; dt2=dt*dt; nConsecLowForceDiff = 0;
      if(dbMode > 2) std::cout << "Saddle Point:  ** Consecutive low dForce => dt = " << dt << std::endl;
    }

    //Shouldn't be necessary since grad_z == 0, but just to be sure all points 
    //  are locked to their given (initial) z-coordinate
    if(bLockToPlane && numDims > 2) {
      for(std::size_t i=1; i<nImagePts-1; i++) force[i][2] = 0.0;
    }

    //Reduce dt if movement of any point exceeds initial average distance btwn pts
    bool reduce_dt = true;
    while(reduce_dt && dt/2 > minTimeStep) {
      reduce_dt = false;
      for(std::size_t i=1; i<nImagePts-1; i++) {
	if(imagePts[i].velocity.norm() * dt > max_dCoords) { reduce_dt = true; break; }
	if(0.5 * force[i].norm() * dt2 > max_dCoords) { reduce_dt = true; break; }
      }
      if(reduce_dt) { 
	dt /= 2; dt2=dt*dt;
	if(dbMode > 2) std::cout << "Saddle Point:  ** Warning: dCoords too large: dt => " << dt << std::endl;
      }
    }
	  
    // adjust image point size based on weight (if requested)
    //  --> try to get weight between MIN/MAX target weights by varying image pt size
    if(bAdaptivePointSize) {
      for(std::size_t i=0; i<nImagePts; i++) {
	if(imagePts[i].weight < minAdaptivePointWt) imagePts[i].radius *= 2;
	else if(imagePts[i].weight > maxAdaptivePointWt) imagePts[i].radius /= 2;
      }
    }

  }  // end of NEB iteration loops

  //deallocate storage used for MPI communication
  delete [] globalPtValues; 
  delete [] globalPtWeights;
  delete [] globalPtGrads;  

  // Check if converged: nIters < maxIters ?
  if(dbMode) {
    if(nIters <= maxIterations) 
      std::cout << "Saddle Point:  NEB Converged after " << nIters << " iterations" << std::endl;
    else
      std::cout << "Saddle Point:  NEB Giving up after " << maxIterations << " iterations" << std::endl;

    for(std::size_t i=0; i<nImagePts; i++) {
      std::cout << "Saddle Point:  --   Final pt[" << i << "] = " << imagePts[i].value 
		<< " : " << imagePts[i].coords << "  (wt = " << imagePts[i].weight << " )" 
		<< "  (r= " << imagePts[i].radius << " )" << std::endl;
    }
  }

  // Choose image point with highest value (and positive weight) as saddle point
  std::size_t imax = 0;
  for(std::size_t i=0; i<nImagePts; i++) {
    if(imagePts[i].weight > 0) { imax = i; break; }
  }
  for(std::size_t i=imax+1; i<nImagePts; i++) {
    if(imagePts[i].value > imagePts[imax].value && imagePts[i].weight > 0) imax = i;
  }
  iSaddlePt = imax;

  if(debugFilename.length() > 0) fDebug.close();
  return;
}
#endif

void
QCAD::SaddleValueResponseFunction::
doNudgedElasticBandT(const double current_time,
		    const Tpetra_Vector* xdotT,
		    const Tpetra_Vector& xT,
		    const Teuchos::Array<ParamVec>& p,
		    Tpetra_Vector& gT, int dbMode)
{
  //  2) Perform Nudged Elastic Band Algorithm to find saddle point.
  //      Iterate over field manager fills of each image point's value and gradient
  //       then update image point positions user Verlet algorithm

  std::size_t nIters, nInitialIterations;
  double dp, s;
  double gradScale, springScale, springBase;
  double avgForce=0, avgOpposingForce=0;
  double dt = maxTimeStep;
  double dt2 = dt*dt;
  double acceptedHighestPtGradNorm = -1.0, highestPtGradNorm;
  int iHighestPt, nConsecLowForceDiff=0;  

  mathVector tangent(numDims);  
  std::vector<mathVector> force(nImagePts), lastForce(nImagePts), lastPositions(nImagePts), lastVelocities(nImagePts);
  std::vector<double> springConstants(nImagePts-1, minSpringConstant);

  //initialize force variables and last positions
  for(std::size_t i=0; i<nImagePts; i++) {
    force[i].resize(numDims); force[i].fill(0.0);
    lastForce[i] = force[i];
    lastPositions[i] = imagePts[i].coords;
    lastVelocities[i] = imagePts[i].velocity;
  }

  //get distance between initial and final points
  double max_dCoords = imagePts[0].coords.distanceTo( imagePts[nImagePts-1].coords ) / nImagePts;

  nIters = 0;
  nInitialIterations = 20; // TODO: make into parameter?
  
  //Storage for aggrecated image point data (needed for MPI)
  double*  globalPtValues   = new double [nImagePts];
  double*  globalPtWeights  = new double [nImagePts];
  double*  globalPtGrads    = new double [nImagePts*numDims];

  //Write headers to output files
  std::fstream fDebug;

  if( outputFilename.length() > 0) {
    std::fstream out; double pathLength = 0.0;
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << std::endl << std::endl << "% Saddle point path - Image points" << std::endl;
    out << "% index xCoord yCoord value pathLength pointRadius" << std::endl;
    for(std::size_t i=0; i<nImagePts; i++) {
      out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	  << " " << imagePts[i].value << " " << pathLength << " " << imagePts[i].radius << std::endl;
      if(i<nImagePts-1) pathLength += imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    }    
    out.close();
  }
  if(debugFilename.length() > 0) {
    fDebug.open(debugFilename.c_str(), std::fstream::out);
    fDebug << "# HighestValue  HighestIndex  AverageForce  TimeStep"
	   << "  HighestPtGradNorm  AverageOpposingForce  SpringBase" << std::endl;
  }



  // Begin NEB iteration loop
  while( ++nIters <= maxIterations) {

    if(dbMode > 1) std::cout << "Saddle Point:  NEB Algorithm iteration " << nIters << " -----------------------" << std::endl;
    writeOutput(nIters);


    if(nIters > 1) {
      //Update coordinates and velocity using (modified) Verlet integration. Reset
      // the velocity to zero if it is directed opposite to force (reduces overshoot)
      for(std::size_t i=1; i<nImagePts-1; i++) {
	dp = imagePts[i].velocity.dot(force[i]);
	if(dp < 0) imagePts[i].velocity.fill(0.0);

	//save last position & velocity in case the new position brings 
	//  us outside the mesh or we need to backtrace
	lastPositions[i] = imagePts[i].coords; 
	lastVelocities[i] = imagePts[i].velocity; //Note: will be zero if force opposed velocity (above)

	// ** Update **
	mathVector dCoords = lastVelocities[i] * dt + force[i] * dt2 * 0.5;
	imagePts[i].coords = lastPositions[i] + dCoords;
	imagePts[i].velocity = lastVelocities[i] + force[i] * dt;
      }
    }

    getImagePointValuesT(current_time, xdotT, xT, p, gT, 
			globalPtValues, globalPtWeights, globalPtGrads,
			lastPositions, dbMode);
    iHighestPt = getHighestPtIndex();
    highestPtGradNorm = imagePts[iHighestPt].grad.norm();

    // Setup scaling factors on first iteration
    if(nIters == 1) initialIterationSetup(gradScale, springScale, dbMode);

    // If in "backtrace" mode, require grad norm to decrease (or take minimum timestep)
    if(nIters > backtraceAfterIters) {
      while(dt > minTimeStep && highestPtGradNorm > acceptedHighestPtGradNorm) {

	//reduce dt
	dt = (dt/2 < minTimeStep) ? minTimeStep : dt/2;
	dt /= 2; dt2=dt*dt; 
	
	if(dbMode > 2) std::cout << "Saddle Point:  ** Backtrace dt => " << dt << std::endl;

	//Update coordinates and velocity using (modified) Verlet integration. Reset
	// the velocity to zero if it is directed opposite to force (reduces overshoot)
	for(std::size_t i=1; i<nImagePts-1; i++) {
	  mathVector dCoords = lastVelocities[i] * dt + force[i] * dt2 * 0.5;
	  imagePts[i].coords = lastPositions[i] + dCoords;
	  imagePts[i].velocity = lastVelocities[i] + force[i] * dt;
	}
	
	getImagePointValuesT(current_time, xdotT, xT, p, gT, 
			    globalPtValues, globalPtWeights, globalPtGrads,
			    lastPositions, dbMode);
	iHighestPt = getHighestPtIndex();
	highestPtGradNorm = imagePts[iHighestPt].grad.norm();
      }

      if(dt == minTimeStep && highestPtGradNorm > acceptedHighestPtGradNorm && dbMode > 2)
	std::cout << "Saddle Point:  ** Warning: backtrace hit min dt == " << dt << std::endl;
    }	
    acceptedHighestPtGradNorm = highestPtGradNorm;

    // Compute spring base constant for this iteration
    s = ((double)nIters-1.0)/maxIterations;    
    springBase = springScale * ( (1.0-s)*minSpringConstant + s*maxSpringConstant ); 
    for(std::size_t i=0; i<nImagePts-1; i++) springConstants[i] = springBase;
	  
    avgForce = avgOpposingForce = 0.0;

    // Compute force acting on each image point
    for(std::size_t i=1; i<nImagePts-1; i++) {
      if(dbMode > 2) std::cout << std::endl << "Saddle Point:  >> Updating pt[" << i << "]:" << imagePts[i];

      // compute the tangent vector for the ith image point
      computeTangent(i, tangent, dbMode);

      // compute the force vector for the ith image point
      if((int)i == iHighestPt && bClimbing && nIters > nInitialIterations)
	computeClimbingForce(i, tangent, gradScale, force[i], dbMode);
      else
	computeForce(i, tangent, springConstants, gradScale, springScale,
		     force[i], dt, dt2, dbMode);

      // update avgForce and avgOpposingForce
      avgForce += force[i].norm();
      // Handle the 0 case so we don't divide by 0.
      const double lastForce_norm = lastForce[i].norm();
      dp = lastForce_norm == 0 ? 0 :
        force[i].dot(lastForce[i]) / (force[i].norm() * lastForce_norm);
      if( dp < 0 ) {  //if current force and last force point in "opposite" directions
	mathVector v = force[i] - lastForce[i];
	avgOpposingForce += v.norm() / (force[i].norm() + lastForce[i].norm());
	//avgOpposingForce += dp;  //an alternate implementation
      } 
    } // end of loop over image points 

    avgForce /= (nImagePts-2);
    avgOpposingForce /= (nImagePts-2);


    //print debug output
    if(dbMode > 1) 
      std::cout << "Saddle Point:  ** Summary:"
		<< "  highest val[" << iHighestPt << "] = " << imagePts[iHighestPt].value
		<< "  AverageForce = " << avgForce << "  dt = " << dt 
		<< "  gradNorm = " << imagePts[iHighestPt].grad.norm() 
		<< "  AvgOpposingForce = " << avgOpposingForce 
		<< "  SpringBase = " << springBase << std::endl;
    if(debugFilename.length() > 0)
      fDebug << imagePts[iHighestPt].value << "  " << iHighestPt << "  "
	     << avgForce << "  "  << dt << "  " << imagePts[iHighestPt].grad.norm() << "  "
	     << avgOpposingForce << "  " << springBase << std::endl;


    // Check for convergence in gradient norm
    if(imagePts[iHighestPt].grad.norm() < convergeTolerance) {
      if(dbMode > 2) std::cout << "Saddle Point:  ** Converged (grad norm " << 
	       imagePts[iHighestPt].grad.norm() << " < " << convergeTolerance << ")" << std::endl;
      break; // exit iterations loop
    }
    else if(nIters == maxIterations) break; //max iterations reached -- exit iterations loop now
                                            // (important so coords & radii don't get updated)

    // Save last force for next iteration
    for(std::size_t i=1; i<nImagePts-1; i++) lastForce[i] = force[i];
    
    // If all forces have remained in the same direction, tally this.  If this happens too many times
    //  increase dt, as this is a sign the time step is too small.
    if(avgOpposingForce < 1e-6) nConsecLowForceDiff++; else nConsecLowForceDiff = 0;
    if(nConsecLowForceDiff >= 3 && dt < maxTimeStep) { 
      dt *= 2; dt2=dt*dt; nConsecLowForceDiff = 0;
      if(dbMode > 2) std::cout << "Saddle Point:  ** Consecutive low dForce => dt = " << dt << std::endl;
    }

    //Shouldn't be necessary since grad_z == 0, but just to be sure all points 
    //  are locked to their given (initial) z-coordinate
    if(bLockToPlane && numDims > 2) {
      for(std::size_t i=1; i<nImagePts-1; i++) force[i][2] = 0.0;
    }

    //Reduce dt if movement of any point exceeds initial average distance btwn pts
    bool reduce_dt = true;
    while(reduce_dt && dt/2 > minTimeStep) {
      reduce_dt = false;
      for(std::size_t i=1; i<nImagePts-1; i++) {
	if(imagePts[i].velocity.norm() * dt > max_dCoords) { reduce_dt = true; break; }
	if(0.5 * force[i].norm() * dt2 > max_dCoords) { reduce_dt = true; break; }
      }
      if(reduce_dt) { 
	dt /= 2; dt2=dt*dt;
	if(dbMode > 2) std::cout << "Saddle Point:  ** Warning: dCoords too large: dt => " << dt << std::endl;
      }
    }
	  
    // adjust image point size based on weight (if requested)
    //  --> try to get weight between MIN/MAX target weights by varying image pt size
    if(bAdaptivePointSize) {
      for(std::size_t i=0; i<nImagePts; i++) {
	if(imagePts[i].weight < minAdaptivePointWt) imagePts[i].radius *= 2;
	else if(imagePts[i].weight > maxAdaptivePointWt) imagePts[i].radius /= 2;
      }
    }

  }  // end of NEB iteration loops

  //deallocate storage used for MPI communication
  delete [] globalPtValues; 
  delete [] globalPtWeights;
  delete [] globalPtGrads;  

  // Check if converged: nIters < maxIters ?
  if(dbMode) {
    if(nIters <= maxIterations) 
      std::cout << "Saddle Point:  NEB Converged after " << nIters << " iterations" << std::endl;
    else
      std::cout << "Saddle Point:  NEB Giving up after " << maxIterations << " iterations" << std::endl;

    for(std::size_t i=0; i<nImagePts; i++) {
      std::cout << "Saddle Point:  --   Final pt[" << i << "] = " << imagePts[i].value 
		<< " : " << imagePts[i].coords << "  (wt = " << imagePts[i].weight << " )" 
		<< "  (r= " << imagePts[i].radius << " )" << std::endl;
    }
  }

  // Choose image point with highest value (and positive weight) as saddle point
  std::size_t imax = 0;
  for(std::size_t i=0; i<nImagePts; i++) {
    if(imagePts[i].weight > 0) { imax = i; break; }
  }
  for(std::size_t i=imax+1; i<nImagePts; i++) {
    if(imagePts[i].value > imagePts[imax].value && imagePts[i].weight > 0) imax = i;
  }
  iSaddlePt = imax;

  if(debugFilename.length() > 0) fDebug.close();

  return;
}

#if defined(ALBANY_EPETRA)
void
QCAD::SaddleValueResponseFunction::
fillSaddlePointData(const double current_time,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    Epetra_Vector& g, int dbMode)
{
  if(dbMode > 1) std::cout << "Saddle Point:  Begin filling saddle point data" << std::endl;

  if(bAggregateWorksets) {  //in aggregate workset mode, there are currently no x-y cutoffs imposed on points in getImagePointValues
    xmin = -1e10; xmax = 1e10; ymin = -1e10; ymax = 1e10;
  }

  mode = "Fill saddle point";
  
  Teuchos::RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);

  //convert xdot_poisson and x_poisson to Tpetra  
  Teuchos::RCP<const Tpetra_Vector> xT, xdotT; 
  xT  = Petra::EpetraVector_To_TpetraVectorConst(x, commT); 
  if (xdot != NULL)  
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  Teuchos::RCP<Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorNonConst(g, commT); 

  Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				   current_time, xdotT.get(), NULL, *xT, p, *gT);
  if(dbMode > 1) std::cout << "Saddle Point:  Done filling saddle point data" << std::endl;

  //Note: MPI: saddle weight is already summed in evaluator's 
  //   postEvaluate, so no need to do anything here

  returnFieldVal = g[0];
  imagePts[iSaddlePt].value = g[1];
  
  // Overwrite response indices 2+ with saddle point coordinates
  for(std::size_t i=0; i<numDims; i++) g[2+i] = imagePts[iSaddlePt].coords[i]; 

  if(dbMode) {
    std::cout << "Saddle Point:  Return Field value = " << g[0] << std::endl;
    std::cout << "Saddle Point:         Field value = " << g[1] << std::endl;
    for(std::size_t i=0; i<numDims; i++)
      std::cout << "Saddle Point:         Coord[" << i << "] = " << g[2+i] << std::endl;
  }

  if( outputFilename.length() > 0) {
    std::fstream out; double pathLength = 0.0;
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << "# Final image points" << std::endl;
    for(std::size_t i=0; i<nImagePts; i++) {
      out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	  << " " << imagePts[i].value << " " << pathLength << " " << imagePts[i].radius << std::endl;
      if(i<nImagePts-1) pathLength += imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    }    
    out.close();
  }

  return;
}
#endif

void
QCAD::SaddleValueResponseFunction::
fillSaddlePointDataT(const double current_time,
		    const Tpetra_Vector* xdotT,
		    const Tpetra_Vector& xT,
		    const Teuchos::Array<ParamVec>& p,
		    Tpetra_Vector& gT, int dbMode)
{
/*
  if(dbMode > 1) std::cout << "Saddle Point:  Begin filling saddle point data" << std::endl;

  if(bAggregateWorksets) {  //in aggregate workset mode, there are currently no x-y cutoffs imposed on points in getImagePointValues
    xmin = -1e10; xmax = 1e10; ymin = -1e10; ymax = 1e10;
  }

  mode = "Fill saddle point";
  Albany::FieldManagerScalarResponseFunction::evaluateResponse(
				   current_time, xdot, x, p, g);
  if(dbMode > 1) std::cout << "Saddle Point:  Done filling saddle point data" << std::endl;

  //Note: MPI: saddle weight is already summed in evaluator's 
  //   postEvaluate, so no need to do anything here

  returnFieldVal = g[0];
  imagePts[iSaddlePt].value = g[1];
  
  // Overwrite response indices 2+ with saddle point coordinates
  for(std::size_t i=0; i<numDims; i++) g[2+i] = imagePts[iSaddlePt].coords[i]; 

  if(dbMode) {
    std::cout << "Saddle Point:  Return Field value = " << g[0] << std::endl;
    std::cout << "Saddle Point:         Field value = " << g[1] << std::endl;
    for(std::size_t i=0; i<numDims; i++)
      std::cout << "Saddle Point:         Coord[" << i << "] = " << g[2+i] << std::endl;
  }

  if( outputFilename.length() > 0) {
    std::fstream out; double pathLength = 0.0;
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << "# Final image points" << std::endl;
    for(std::size_t i=0; i<nImagePts; i++) {
      out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	  << " " << imagePts[i].value << " " << pathLength << " " << imagePts[i].radius << std::endl;
      if(i<nImagePts-1) pathLength += imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    }    
    out.close();
  }
*/
  return;
}

#if defined(ALBANY_EPETRA)
void
QCAD::SaddleValueResponseFunction::
doLevelSet(const double current_time,
	   const Epetra_Vector* xdot,
	   const Epetra_Vector& x,
	   const Teuchos::Array<ParamVec>& p,
	   Epetra_Vector& g, int dbMode)
{
  int result;
  const Epetra_Comm& comm = x.Map().Comm();

  if( fabs(levelSetRadius) < 1e-9 ) return; //don't run if level-set radius is zero

  vlsFieldValues.clear(); vlsCellAreas.clear();
  for(std::size_t k=0; k<numDims; k++) vlsCoords[k].clear();

  mode = "Level set data collection";
  
  Teuchos::RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);

  //convert xdot_poisson and x_poisson to Tpetra  
  Teuchos::RCP<const Tpetra_Vector> xT, xdotT; 
  xT  = Petra::EpetraVector_To_TpetraVectorConst(x, commT); 
  if (xdot != NULL)  
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
  Teuchos::RCP<Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorNonConst(g, commT); 

  Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				   current_time, xdotT.get(), NULL, *xT, p, *gT);

  //! Gather data from different processors
  std::vector<double> allFieldVals;
  std::vector<double> allCellAreas;
  std::vector<double> allCoords[MAX_DIMENSIONS];

  QCAD::gatherVector(vlsFieldValues, allFieldVals, comm);  
  QCAD::gatherVector(vlsCellAreas, allCellAreas, comm);
  for(std::size_t k=0; k<numDims; k++)
    QCAD::gatherVector(vlsCoords[k], allCoords[k], comm);

  //! Exit early if there are no field values in the specified region
  if( allFieldVals.size()  == 0 ) return;

  //! Print gathered size on proc 0
  if(dbMode) {
    std::cout << std::endl << "--- Begin Saddle Level Set Algorithm ---" << std::endl;
    std::cout << "--- Saddle Level Set: local size (this proc) = " << vlsFieldValues.size()
	      << ", gathered size (all procs) = " << allFieldVals.size() << std::endl;
  }

  //! Sort data by field value
  std::vector<int> ordering;
  QCAD::getOrdering(allFieldVals, ordering);


  //! Compute max/min field values
  double maxFieldVal = allFieldVals[0], minFieldVal = allFieldVals[0];
  double maxCoords[3], minCoords[3];

  for(std::size_t k=0; k<numDims && k < 3; k++)
    maxCoords[k] = minCoords[k] = allCoords[k][0];

  std::size_t N = allFieldVals.size();
  for(std::size_t i=0; i<N; i++) {
    for(std::size_t k=0; k<numDims && k < 3; k++) {
      if(allCoords[k][i] > maxCoords[k]) maxCoords[k] = allCoords[k][i];
      if(allCoords[k][i] < minCoords[k]) minCoords[k] = allCoords[k][i];
    }
    if(allFieldVals[i] > maxFieldVal) maxFieldVal = allFieldVals[i];
    if(allFieldVals[i] < minFieldVal) minFieldVal = allFieldVals[i];
  }
  
  double avgCellLength = pow(QCAD::averageOfVector(allCellAreas), 0.5); //assume 2D areas
  double maxFieldDifference = fabs(maxFieldVal - minFieldVal);
  double currentSaddleValue = imagePts[iSaddlePt].value;
  

  if(dbMode > 1) {
    std::cout << "--- Saddle Level Set: max field difference = " << maxFieldDifference
	      << ", avg cell length = " << avgCellLength << std::endl;
  }

  //! Set cutoffs
  double cutoffDistance, cutoffFieldVal, minDepth;
  cutoffDistance = avgCellLength * distanceCutoffFctr;
  cutoffFieldVal = maxFieldDifference * fieldCutoffFctr;
  minDepth = minPoolDepthFctr * (currentSaddleValue - minFieldVal) / 2.0; //maxFieldDifference * minPoolDepthFctr;

  result = FindSaddlePoint_LevelSet(allFieldVals, allCoords, ordering,
			   cutoffDistance, cutoffFieldVal, minDepth, dbMode, g);
  // result == 0 ==> success: found 2 "deep" pools & saddle pt
  if(result == 0) { 
    //update imagePts[iSaddlePt] to be newly found saddle value
    for(std::size_t i=0; i<numDims; i++) imagePts[iSaddlePt].coords[i] = g[2+i];
    imagePts[iSaddlePt].value = g[1];
    imagePts[iSaddlePt].radius = 1e-5; //very small so only pick up point of interest?
    //set weight?
  }

  return;
}
#endif

void
QCAD::SaddleValueResponseFunction::
doLevelSetT(const double current_time,
	   const Tpetra_Vector* xdotT,
	   const Tpetra_Vector& xT,
	   const Teuchos::Array<ParamVec>& p,
	   Tpetra_Vector& gT, int dbMode)
{

  int result;
  Teuchos::RCP<const Teuchos::Comm<int> > commT = xT.getMap()->getComm();

  if( fabs(levelSetRadius) < 1e-9 ) return; //don't run if level-set radius is zero

  vlsFieldValues.clear(); vlsCellAreas.clear();
  for(std::size_t k=0; k<numDims; k++) vlsCoords[k].clear();

  mode = "Level set data collection";
  Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				   current_time, xdotT, NULL, xT, p, gT);

  //! Gather data from different processors
  std::vector<double> allFieldVals;
  std::vector<double> allCellAreas;
  std::vector<double> allCoords[MAX_DIMENSIONS];

  QCAD::gatherVectorT(vlsFieldValues, allFieldVals, commT);  
  QCAD::gatherVectorT(vlsCellAreas, allCellAreas, commT);
  for(std::size_t k=0; k<numDims; k++)
    QCAD::gatherVectorT(vlsCoords[k], allCoords[k], commT);

  //! Exit early if there are no field values in the specified region
  if( allFieldVals.size()  == 0 ) return;

  //! Print gathered size on proc 0
  if(dbMode) {
    std::cout << std::endl << "--- Begin Saddle Level Set Algorithm ---" << std::endl;
    std::cout << "--- Saddle Level Set: local size (this proc) = " << vlsFieldValues.size()
	      << ", gathered size (all procs) = " << allFieldVals.size() << std::endl;
  }

  //! Sort data by field value
  std::vector<int> ordering;
  QCAD::getOrdering(allFieldVals, ordering);


  //! Compute max/min field values
  double maxFieldVal = allFieldVals[0], minFieldVal = allFieldVals[0];
  double maxCoords[3], minCoords[3];

  for(std::size_t k=0; k<numDims && k < 3; k++)
    maxCoords[k] = minCoords[k] = allCoords[k][0];

  std::size_t N = allFieldVals.size();
  for(std::size_t i=0; i<N; i++) {
    for(std::size_t k=0; k<numDims && k < 3; k++) {
      if(allCoords[k][i] > maxCoords[k]) maxCoords[k] = allCoords[k][i];
      if(allCoords[k][i] < minCoords[k]) minCoords[k] = allCoords[k][i];
    }
    if(allFieldVals[i] > maxFieldVal) maxFieldVal = allFieldVals[i];
    if(allFieldVals[i] < minFieldVal) minFieldVal = allFieldVals[i];
  }
  
  double avgCellLength = pow(QCAD::averageOfVector(allCellAreas), 0.5); //assume 2D areas
  double maxFieldDifference = fabs(maxFieldVal - minFieldVal);
  double currentSaddleValue = imagePts[iSaddlePt].value;
  

  if(dbMode > 1) {
    std::cout << "--- Saddle Level Set: max field difference = " << maxFieldDifference
	      << ", avg cell length = " << avgCellLength << std::endl;
  }

  //! Set cutoffs
  double cutoffDistance, cutoffFieldVal, minDepth;
  cutoffDistance = avgCellLength * distanceCutoffFctr;
  cutoffFieldVal = maxFieldDifference * fieldCutoffFctr;
  minDepth = minPoolDepthFctr * (currentSaddleValue - minFieldVal) / 2.0; //maxFieldDifference * minPoolDepthFctr;

  result = FindSaddlePoint_LevelSetT(allFieldVals, allCoords, ordering,
			   cutoffDistance, cutoffFieldVal, minDepth, dbMode, gT);
  Teuchos::ArrayRCP<const ST> gT_constView = gT.get1dView();
  // result == 0 ==> success: found 2 "deep" pools & saddle pt
  if(result == 0) { 
    //update imagePts[iSaddlePt] to be newly found saddle value
    for(std::size_t i=0; i<numDims; i++) imagePts[iSaddlePt].coords[i] = gT_constView[2+i];
    imagePts[iSaddlePt].value = gT_constView[1];
    imagePts[iSaddlePt].radius = 1e-5; //very small so only pick up point of interest?
    //set weight?
  }

  return;
}

#if defined(ALBANY_EPETRA)
//! Level-set Algorithm for finding saddle point
int QCAD::SaddleValueResponseFunction::
FindSaddlePoint_LevelSet(std::vector<double>& allFieldVals,
		std::vector<double>* allCoords, std::vector<int>& ordering,
		double cutoffDistance, double cutoffFieldVal, double minDepth, int dbMode,
		Epetra_Vector& g)
{
  if(dbMode) {
    std::cout << "--- Saddle Level Set: distance cutoff = " << cutoffDistance
	      << ", field cutoff = " << cutoffFieldVal 
	      << ", min depth = " << minDepth << std::endl;
  }

  // Walk through sorted data.  At current point, walk backward in list 
  //  until either 1) a "close" point is found, as given by tolerance -> join to tree
  //            or 2) the change in field value exceeds some maximium -> new tree
  std::size_t N = allFieldVals.size();
  std::vector<int> treeIDs(N, -1);
  std::vector<double> minFieldVals; //for each tree
  std::vector<int> treeSizes; //for each tree
  int nextAvailableTreeID = 0;

  int nTrees = 0, nMaxTrees = 0;
  int nDeepTrees=0, lastDeepTrees=0, treeIDtoReplace;
  int I, J, K;
  for(std::size_t i=0; i < N; i++) {
    I = ordering[i];

    if(dbMode > 1) {
      nDeepTrees = 0;
      for(std::size_t t=0; t < treeSizes.size(); t++) {
	if(treeSizes[t] > 0 && (allFieldVals[I]-minFieldVals[t]) > minDepth) nDeepTrees++;
      }
    }

    if(dbMode > 3) std::cout << "DEBUG: i=" << i << "( I = " << I << "), val="
			 << allFieldVals[I] << ", loc=(" << allCoords[0][I] 
			 << "," << allCoords[1][I] << ")" << " nD=" << nDeepTrees;

    if(dbMode > 1 && lastDeepTrees != nDeepTrees) {
      std::cout << "--- Saddle: i=" << i << " new deep pool: nPools=" << nTrees 
		<< " nDeep=" << nDeepTrees << std::endl;
      lastDeepTrees = nDeepTrees;
    }

    for(int j=i-1; j >= 0 && fabs(allFieldVals[I] - allFieldVals[ordering[j]]) < cutoffFieldVal; j--) {
      J = ordering[j];

      if( QCAD::distance(allCoords, I, J, numDims) < cutoffDistance ) {
	if(treeIDs[I] == -1) {
	  treeIDs[I] = treeIDs[J];
	  treeSizes[treeIDs[I]]++;

	  if(dbMode > 3) std::cout << " --> tree " << treeIDs[J] 
			       << " ( size=" << treeSizes[treeIDs[J]] << ", depth=" 
			       << (allFieldVals[I]-minFieldVals[treeIDs[J]]) << ")" << std::endl;
	}
	else if(treeIDs[I] != treeIDs[J]) {

	  //update number of deep trees
	  nDeepTrees = 0;
	  for(std::size_t t=0; t < treeSizes.size(); t++) {
	    if(treeSizes[t] > 0 && (allFieldVals[I]-minFieldVals[t]) > minDepth) nDeepTrees++;
	  }

	  bool mergingTwoDeepTrees = false;
	  if((allFieldVals[I]-minFieldVals[treeIDs[I]]) > minDepth && 
	     (allFieldVals[I]-minFieldVals[treeIDs[J]]) > minDepth) {
	    mergingTwoDeepTrees = true;
	    nDeepTrees--;
	  }

	  treeIDtoReplace = treeIDs[I];
	  if( minFieldVals[treeIDtoReplace] < minFieldVals[treeIDs[J]] )
	    minFieldVals[treeIDs[J]] = minFieldVals[treeIDtoReplace];

	  for(int k=i; k >=0; k--) {
	    K = ordering[k];
	    if(treeIDs[K] == treeIDtoReplace) {
	      treeIDs[K] = treeIDs[J];
	      treeSizes[treeIDs[J]]++;
	    }
	  }
	  treeSizes[treeIDtoReplace] = 0;
	  nTrees -= 1;

	  if(dbMode > 3) std::cout << "DEBUG:   also --> " << treeIDs[J] 
			       << " [merged] size=" << treeSizes[treeIDs[J]]
			       << " (treecount after merge = " << nTrees << ")" << std::endl;

	  if(dbMode > 1) std::cout << "--- Saddle: i=" << i << "merge: nPools=" << nTrees 
				   << " nDeep=" << nDeepTrees << std::endl;


	  if(mergingTwoDeepTrees && nDeepTrees == 1) {
	    if(dbMode > 3) std::cout << "DEBUG: FOUND SADDLE! exiting." << std::endl;
	    if(dbMode > 1) std::cout << "--- Saddle: i=" << i << " Found saddle at ";

            //Found saddle at I
	    g[0] = 0; //TODO - change this g[.] interface to something more readable -- and we don't use g[0] now
            g[1] = allFieldVals[I];
            for(std::size_t k=0; k<numDims && k < 3; k++) {
              g[2+k] = allCoords[k][I];
              if(dbMode > 1) std::cout << allCoords[k][I] << ", ";
	    }
            
	    if(dbMode > 1) std::cout << "ret=" << g[0] << std::endl;
            return 0; //success
	  }

	}
      }

    } //end j loop
    
    if(treeIDs[I] == -1) {
      if(dbMode > 3) std::cout << " --> new tree with ID " << nextAvailableTreeID
			       << " (treecount after new = " << (nTrees+1) << ")" << std::endl;
      if(dbMode > 1) std::cout << "--- Saddle: i=" << i << " new pool: nPools=" << (nTrees+1) 
			       << " nDeep=" << nDeepTrees << std::endl;

      treeIDs[I] = nextAvailableTreeID++;
      minFieldVals.push_back(allFieldVals[I]);
      treeSizes.push_back(1);

      nTrees += 1;
      if(nTrees > nMaxTrees) nMaxTrees = nTrees;
    }

  } // end i loop

  // if no saddle found, return all zeros
  if(dbMode > 3) std::cout << "DEBUG: NO SADDLE. exiting." << std::endl;
  for(std::size_t k=0; k<5; k++) g[k] = 0;

  // if two or more trees where found, then reason for failure is that not
  //  enough deep pools were found - so could try to reduce minDepth and re-run.
  if(nMaxTrees >= 2) return 1;

  // nMaxTrees < 2 - so we need more trees.  Could try to increase cutoffDistance and/or cutoffFieldVal.
  return 2;
}
#endif

int QCAD::SaddleValueResponseFunction::
FindSaddlePoint_LevelSetT(std::vector<double>& allFieldVals,
		std::vector<double>* allCoords, std::vector<int>& ordering,
		double cutoffDistance, double cutoffFieldVal, double minDepth, int dbMode,
		Tpetra_Vector& gT)
{
  Teuchos::ArrayRCP<ST> gT_nonconstView;
  if(dbMode) {
    std::cout << "--- Saddle Level Set: distance cutoff = " << cutoffDistance
	      << ", field cutoff = " << cutoffFieldVal 
	      << ", min depth = " << minDepth << std::endl;
  }

  // Walk through sorted data.  At current point, walk backward in list 
  //  until either 1) a "close" point is found, as given by tolerance -> join to tree
  //            or 2) the change in field value exceeds some maximium -> new tree
  std::size_t N = allFieldVals.size();
  std::vector<int> treeIDs(N, -1);
  std::vector<double> minFieldVals; //for each tree
  std::vector<int> treeSizes; //for each tree
  int nextAvailableTreeID = 0;

  int nTrees = 0, nMaxTrees = 0;
  int nDeepTrees=0, lastDeepTrees=0, treeIDtoReplace;
  int I, J, K;
  for(std::size_t i=0; i < N; i++) {
    I = ordering[i];

    if(dbMode > 1) {
      nDeepTrees = 0;
      for(std::size_t t=0; t < treeSizes.size(); t++) {
	if(treeSizes[t] > 0 && (allFieldVals[I]-minFieldVals[t]) > minDepth) nDeepTrees++;
      }
    }

    if(dbMode > 3) std::cout << "DEBUG: i=" << i << "( I = " << I << "), val="
			 << allFieldVals[I] << ", loc=(" << allCoords[0][I] 
			 << "," << allCoords[1][I] << ")" << " nD=" << nDeepTrees;

    if(dbMode > 1 && lastDeepTrees != nDeepTrees) {
      std::cout << "--- Saddle: i=" << i << " new deep pool: nPools=" << nTrees 
		<< " nDeep=" << nDeepTrees << std::endl;
      lastDeepTrees = nDeepTrees;
    }

    for(int j=i-1; j >= 0 && fabs(allFieldVals[I] - allFieldVals[ordering[j]]) < cutoffFieldVal; j--) {
      J = ordering[j];

      if( QCAD::distance(allCoords, I, J, numDims) < cutoffDistance ) {
	if(treeIDs[I] == -1) {
	  treeIDs[I] = treeIDs[J];
	  treeSizes[treeIDs[I]]++;

	  if(dbMode > 3) std::cout << " --> tree " << treeIDs[J] 
			       << " ( size=" << treeSizes[treeIDs[J]] << ", depth=" 
			       << (allFieldVals[I]-minFieldVals[treeIDs[J]]) << ")" << std::endl;
	}
	else if(treeIDs[I] != treeIDs[J]) {

	  //update number of deep trees
	  nDeepTrees = 0;
	  for(std::size_t t=0; t < treeSizes.size(); t++) {
	    if(treeSizes[t] > 0 && (allFieldVals[I]-minFieldVals[t]) > minDepth) nDeepTrees++;
	  }

	  bool mergingTwoDeepTrees = false;
	  if((allFieldVals[I]-minFieldVals[treeIDs[I]]) > minDepth && 
	     (allFieldVals[I]-minFieldVals[treeIDs[J]]) > minDepth) {
	    mergingTwoDeepTrees = true;
	    nDeepTrees--;
	  }

	  treeIDtoReplace = treeIDs[I];
	  if( minFieldVals[treeIDtoReplace] < minFieldVals[treeIDs[J]] )
	    minFieldVals[treeIDs[J]] = minFieldVals[treeIDtoReplace];

	  for(int k=i; k >=0; k--) {
	    K = ordering[k];
	    if(treeIDs[K] == treeIDtoReplace) {
	      treeIDs[K] = treeIDs[J];
	      treeSizes[treeIDs[J]]++;
	    }
	  }
	  treeSizes[treeIDtoReplace] = 0;
	  nTrees -= 1;

	  if(dbMode > 3) std::cout << "DEBUG:   also --> " << treeIDs[J] 
			       << " [merged] size=" << treeSizes[treeIDs[J]]
			       << " (treecount after merge = " << nTrees << ")" << std::endl;

	  if(dbMode > 1) std::cout << "--- Saddle: i=" << i << "merge: nPools=" << nTrees 
				   << " nDeep=" << nDeepTrees << std::endl;


	  if(mergingTwoDeepTrees && nDeepTrees == 1) {
	    if(dbMode > 3) std::cout << "DEBUG: FOUND SADDLE! exiting." << std::endl;
	    if(dbMode > 1) std::cout << "--- Saddle: i=" << i << " Found saddle at ";

            gT_nonconstView = gT.get1dViewNonConst();
            //Found saddle at I
	    gT_nonconstView[0] = 0; //TODO - change this g[.] interface to something more readable -- and we don't use g[0] now
            gT_nonconstView[1] = allFieldVals[I];
            for(std::size_t k=0; k<numDims && k < 3; k++) {
              gT_nonconstView[2+k] = allCoords[k][I];
              if(dbMode > 1) std::cout << allCoords[k][I] << ", ";
	    }
            
	    if(dbMode > 1) std::cout << "ret=" << gT_nonconstView[0] << std::endl;
            return 0; //success
	  }

	}
      }

    } //end j loop
    
    if(treeIDs[I] == -1) {
      if(dbMode > 3) std::cout << " --> new tree with ID " << nextAvailableTreeID
			       << " (treecount after new = " << (nTrees+1) << ")" << std::endl;
      if(dbMode > 1) std::cout << "--- Saddle: i=" << i << " new pool: nPools=" << (nTrees+1) 
			       << " nDeep=" << nDeepTrees << std::endl;

      treeIDs[I] = nextAvailableTreeID++;
      minFieldVals.push_back(allFieldVals[I]);
      treeSizes.push_back(1);

      nTrees += 1;
      if(nTrees > nMaxTrees) nMaxTrees = nTrees;
    }

  } // end i loop

  // if no saddle found, return all zeros
  gT_nonconstView = gT.get1dViewNonConst();
  if(dbMode > 3) std::cout << "DEBUG: NO SADDLE. exiting." << std::endl;
  for(std::size_t k=0; k<5; k++) gT_nonconstView[k] = 0;

  // if two or more trees where found, then reason for failure is that not
  //  enough deep pools were found - so could try to reduce minDepth and re-run.
  if(nMaxTrees >= 2) return 1;

  // nMaxTrees < 2 - so we need more trees.  Could try to increase cutoffDistance and/or cutoffFieldVal.

 return 2;
}


double QCAD::SaddleValueResponseFunction::getCurrent
  (const double& lattTemp, const Teuchos::RCP<Albany::MaterialDatabase>& materialDB) const
{
  const double kB = 8.617332e-5;  // eV/K
  double Temp = lattTemp;   // K

  // segmentLength[i] == distance between imagePt[i] and imagePt[i+1]
  double* segmentLength = new double[nImagePts-1]; 

  // path length before and after saddle point
  double lengthBefore = 0.0, lengthAfter = 0.0;    

  // get the distances along each leg of the final saddle path
  for(std::size_t i = 0; i < nImagePts-1; i++) 
  {
    segmentLength[i] = imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    if( (int)i < iSaddlePt ) 
      lengthBefore += segmentLength[i];
    else 
      lengthAfter += segmentLength[i];
  }
  
  // recalculate gfGridSpacing to obtain an integer number of points for GF-CBR calculation,
  // this strategy produces path length that is slightly non-equally spaced due to precision.
  // int nGFPts = int( (lengthBefore + lengthAfter)/gfGridSpacing ) + 1;
  // double ptSpacing = (lengthBefore + lengthAfter)/(nGFPts-1);  // actual GF Grid Spacing
  
  // set actual grid spacing to user-defined value for GF-CBR calculation,
  double ptSpacing = gfGridSpacing;
  int nGFPts = int( (lengthBefore + lengthAfter)/gfGridSpacing );
  std::cout << "nGFPts = " << nGFPts << ", actual gfGridSpacing = " << ptSpacing << std::endl; 

  // hard-code eff. mass along the saddle path using Reference Material's transverse eff. mass
  std::string refMtrlName = materialDB->getParam<std::string>("Reference Material");
  double effMass = materialDB->getMaterialParam<double>(refMtrlName,"Transverse Electron Effective Mass");

  // hard-code electron affinity using the Reference Material's value
  double matChi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");

  // unscale the Field Scaling Factor to obtain the actual Potential in [V]
  Teuchos::RCP<std::vector<double> > pot = Teuchos::rcp( new std::vector<double>(finalPts.size()) );
  for(std::size_t i = 0; i < finalPts.size(); i++)
    (*pot)[i] = finalPts[i].value / fieldScaling;
    
  // compute energy reference
  double qPhiRef;
  {
    std::string category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
    if (category == "Semiconductor") 
    {
      // Same qPhiRef needs to be used for the entire structure
      double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
      double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
      double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
      double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap");
      double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient");
      double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");
      
      double Eg = Eg0 - alpha*pow(Temp,2.0)/(beta+Temp); // in [eV]
      double kbT = kB * Temp;      // in [eV]
      double Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [eV]
      qPhiRef = Chi - Eic;  // (Evac-Ei) in [eV] where Evac = vacuum level
    }
    else 
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid category " << category << " for reference material !" << std::endl);
  }  
  
  // compute conduction band [eV] from the potential
  Teuchos::RCP<std::vector<double> > Ec = Teuchos::rcp( new std::vector<double>(finalPts.size()) );
  Teuchos::RCP<std::vector<double> > pathLen = Teuchos::rcp( new std::vector<double>(finalPts.size()) );
  
  double pathLength = 0.0;
  for(std::size_t i = 0; i < finalPts.size(); i++)
  {
    (*Ec)[i] = qPhiRef - matChi - (*pot)[i];
    (*pathLen)[i] = pathLength;  
    if (i < (finalPts.size()-1))
      pathLength += finalPts[i].coords.distanceTo(finalPts[i+1].coords);
  }
    
  // append the computed Ec data to output file
  if( outputFilename.length() > 0) 
  {
    std::fstream out; 
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << std::endl << std::endl << "% Computed conduction band Ec (assumes field is Potential) -- 'final' points" << std::endl;
    out << "% index xCoord yCoord Ec pathLength pointRadius" << std::endl;
    for(std::size_t i = 0; i < finalPts.size(); i++) 
    {
      out << i << " " << finalPts[i].coords[0] << " " << finalPts[i].coords[1] 
	        << " " << (*Ec)[i] << " " << (*pathLen)[i] << " " << finalPts[i].radius << std::endl;
    }
    out.close();
  }   

  double I = 0.0; 

  // instantiate the GF-CBR solver to compute current
  QCAD::GreensFunctionTunnelingSolver solver(Ec, pathLen, nGFPts, ptSpacing, effMass, comm, outputFilename); //Teuchos::rcp(comm.Clone())
  
  // set the eigensolver to be used
  bool bUseAnasazi = false; 
  if (gfEigensolver == "Anasazi")
     bUseAnasazi = true; 
  else if (gfEigensolver == "tql2")
     bUseAnasazi = false; 
  else
     TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error!  Invalid GF-CBR Method Eigensolver, must be either Anasazi or tql2 !" << std::endl);
  
  // compute the currents for a range of Vds values
  if (bSweepVds)   
  {
    int ptsVds = stepsVds + 1;  // include the ending point
    std::vector<double> rangeVds(ptsVds, 0.0);
    std::vector<double> rangeIds(ptsVds, 0.0); 
    
    double deltaVds = (finalVds - initVds) / double(ptsVds-1); 
    for (int i = 0; i < ptsVds; i++)
      rangeVds[i] = initVds + deltaVds * double(i); 
    
    solver.computeCurrentRange(rangeVds, kB * Temp, current_Ecutoff_offset_from_Emax, rangeIds, bUseAnasazi);
    
    // set I to the last value of rangeIds
    I = rangeIds[ptsVds-1];

    // write Vds vs. Ids data to file
    if ( outputFilename.length() > 0) 
    {
      std::fstream out; 
      out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
      out << std::endl << std::endl << "% Current vs Voltage IV curve (GF-CBR method)" << std::endl;
      out << "% index, Vds [V] vs Ids [A] data" << std::endl;
      for(std::size_t i = 0; i < rangeVds.size(); i++) 
	      out << i << " " << rangeVds[i] << " " << rangeIds[i] << " " << std::endl;
      out.close();
    }
    
  }  // end of if (bSweepVds)
  
  // compute the current for a single Vds value
  else             
    I = solver.computeCurrent(finalVds, kB * Temp, current_Ecutoff_offset_from_Emax, bUseAnasazi); // overwrite "Return Field Val" with current
  
  std::cout << "Final Vds = " << finalVds << " V, Current Ids = " << I << " Amps" << std::endl;
  
  return I;
}


void
QCAD::SaddleValueResponseFunction::
evaluateTangentT(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
		const Tpetra_Vector* xdotT,
		const Tpetra_Vector* xdotdotT,
		const Tpetra_Vector& xT,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Tpetra_MultiVector* VxdotT,
		const Tpetra_MultiVector* VxdotdotT,
		const Tpetra_MultiVector* VxT,
		const Tpetra_MultiVector* VpT,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  // Require that g be computed to get tangent info 
  if(gT != NULL) {
    
    // HACK: for now do not evaluate response when tangent is requested,
    //   as it is assumed that evaluateResponse has already been called
    //   directly or by evaluateGradient.  This prevents repeated calling 
    //   of evaluateResponse within the dg/dp loop of Albany::ModelEvaluator's
    //   evalModel(...) function.  matchesCurrentResults(...) would be able to 
    //   determine if evaluateResponse needs to be run, but 
    //   Albany::AggregateScalarReponseFunction does not copy from global g 
    //   to local g so the g parameter passed to this function will always 
    //   be zeros when used in an aggregate response fn.  Change this?

    // Evaluate response g and run algorithm (if it hasn't run already)
    //if(!matchesCurrentResults(*g)) 
    //  evaluateResponse(current_time, xdot, x, p, *g);

    mode = "Fill saddle point";
    Albany::FieldManagerScalarResponseFunction::evaluateTangentT(
                alpha, beta, omega, current_time, sum_derivs, xdotT,
  	        xdotdotT, xT, p, deriv_p, VxdotT, VxdotdotT, VxT, VpT, gT, gxT, gpT);
  }
  else {
    if (gxT != NULL) gxT->putScalar(0.0);
    if (gpT != NULL) gpT->putScalar(0.0);
  }
}

void
QCAD::SaddleValueResponseFunction::
evaluateGradientT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Tpetra_Vector* gT,
		 Tpetra_MultiVector* dg_dxT,
		 Tpetra_MultiVector* dg_dxdotT,
		 Tpetra_MultiVector* dg_dxdotdotT,
		 Tpetra_MultiVector* dg_dpT)
{
  // Require that g be computed to get gradient info 
  if(gT != NULL) {

    // Evaluate response g and run algorithm (if it hasn't run already)
    if(!matchesCurrentResultsT(*gT)) 
      evaluateResponseT(current_time, xdotT, NULL, xT, p, *gT);

    mode = "Fill saddle point";
    Albany::FieldManagerScalarResponseFunction::evaluateGradientT(
	   current_time, xdotT, xdotdotT, xT, p, deriv_p, gT, dg_dxT, dg_dxdotT, dg_dxdotdotT, dg_dpT);
  }
  else {
    if (dg_dxT != NULL)    dg_dxT->putScalar(0.0);
    if (dg_dxdotT != NULL) dg_dxdotT->putScalar(0.0);
    if (dg_dpT != NULL)    dg_dpT->putScalar(0.0);
  }

}

#if defined(ALBANY_EPETRA)
//IK, 10/9/14: are these functions even needed...
void 
QCAD::SaddleValueResponseFunction::
postProcessResponses(const Epetra_Comm& comm_, const Teuchos::RCP<Epetra_Vector>& g)
{
}

void 
QCAD::SaddleValueResponseFunction::
postProcessResponseDerivatives(const Epetra_Comm& comm_, const Teuchos::RCP<Epetra_MultiVector>& gt)
{
}

void
QCAD::SaddleValueResponseFunction::
getImagePointValues(const double current_time,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    Epetra_Vector& g, 
		    double* globalPtValues,
		    double* globalPtWeights,
		    double* globalPtGrads,
		    std::vector<QCAD::mathVector> lastPositions,
		    int dbMode)
{
  //Set xmax,xmin,ymax,ymin based on points
  xmax = xmin = imagePts[0].coords[0];
  ymax = ymin = imagePts[0].coords[1];
  for(std::size_t i=1; i<nImagePts; i++) {
    xmin = std::min(imagePts[i].coords[0],xmin);
    xmax = std::max(imagePts[i].coords[0],xmax);
    ymin = std::min(imagePts[i].coords[1],ymin);
    ymax = std::max(imagePts[i].coords[1],ymax);
  }
  xmin -= 5*imagePtSize; xmax += 5*imagePtSize;
  ymin -= 5*imagePtSize; ymax += 5*imagePtSize;
    
  //Reset value, weight, and gradient of image points as these are accumulated by evaluator fill
  imagePtValues.fill(0.0);
  imagePtWeights.fill(0.0);
  imagePtGradComps.fill(0.0);

  if(bAggregateWorksets) {
    //Use cached field and coordinate values to perform fill    
    for(std::size_t i=0; i<vFieldValues.size(); i++) {
      addImagePointData( vCoords[i].data, vFieldValues[i], vGrads[i].data );
    } 
  }
  else {
    mode = "Collect image point data";
    Teuchos::RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);

    //convert xdot_poisson and x_poisson to Tpetra  
    Teuchos::RCP<const Tpetra_Vector> xT, xdotT; 
    xT  = Petra::EpetraVector_To_TpetraVectorConst(x, commT); 
    if (xdot != NULL)  
      xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
    Teuchos::RCP<Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorNonConst(g, commT); 

    Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				     current_time, xdotT.get(), NULL, *xT, p, *gT);
  }

  //MPI -- sum weights, value, and gradient for each image pt
  comm->SumAll( imagePtValues.data(),    globalPtValues,  nImagePts );
  comm->SumAll( imagePtWeights.data(),   globalPtWeights, nImagePts );
  comm->SumAll( imagePtGradComps.data(), globalPtGrads,   nImagePts*numDims );

  // Put summed data into imagePts, normalizing value and 
  //   gradient from different cell contributions
  for(std::size_t i=0; i<nImagePts; i++) {
    imagePts[i].weight = globalPtWeights[i];
    if(globalPtWeights[i] > 1e-6) {
      imagePts[i].value = globalPtValues[i] / imagePts[i].weight;
      for(std::size_t k=0; k<numDims; k++) 
	imagePts[i].grad[k] = globalPtGrads[k*nImagePts+i] / imagePts[i].weight;
    }
    else { 
      //assume point has drifted off region: leave value as, set gradient to zero, and reset position
      imagePts[i].grad.fill(0.0);
      imagePts[i].coords = lastPositions[i];
    }
  }

  return;
}

void
QCAD::SaddleValueResponseFunction::
getFinalImagePointValues(const double current_time,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    Epetra_Vector& g, 
		    int dbMode)
{
  //Set xmax,xmin,ymax,ymin based on points
  xmax = xmin = imagePts[0].coords[0];
  ymax = ymin = imagePts[0].coords[1];
  for(std::size_t i=1; i<nImagePts; i++) {
    xmin = std::min(imagePts[i].coords[0],xmin);
    xmax = std::max(imagePts[i].coords[0],xmax);
    ymin = std::min(imagePts[i].coords[1],ymin);
    ymax = std::max(imagePts[i].coords[1],ymax);
  }
  xmin -= 5*imagePtSize; xmax += 5*imagePtSize;
  ymin -= 5*imagePtSize; ymax += 5*imagePtSize;
    
  //Reset value and weight of final image points as these are accumulated by evaluator fill
  finalPtValues.fill(0.0);
  finalPtWeights.fill(0.0);

  if(bAggregateWorksets) {
    //Use cached field and coordinate values to perform fill    
    for(std::size_t i=0; i<vFieldValues.size(); i++) {
      addFinalImagePointData( vCoords[i].data, vFieldValues[i] );
    } 
  }
  else {
    mode = "Collect final image point data";
    Teuchos::RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);

    //convert xdot_poisson and x_poisson to Tpetra  
    Teuchos::RCP<const Tpetra_Vector> xT, xdotT; 
    xT  = Petra::EpetraVector_To_TpetraVectorConst(x, commT); 
    if (xdot != NULL)  
      xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
    Teuchos::RCP<Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorNonConst(g, commT); 

    Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				     current_time, xdotT.get(), NULL, *xT, p, *gT);
  }

  //MPI -- sum weights, value, and gradient for each image pt
  std::size_t nFinalPts = finalPts.size();
  if(nFinalPts > 0) {
    double*  globalPtValues   = new double [nFinalPts];
    double*  globalPtWeights  = new double [nFinalPts];
    comm->SumAll( finalPtValues.data(),    globalPtValues,  nFinalPts );
    comm->SumAll( finalPtWeights.data(),   globalPtWeights, nFinalPts );

    // Put summed data into imagePts, normalizing value from different cell contributions
    for(std::size_t i=0; i<nFinalPts; i++) {
      finalPts[i].weight = globalPtWeights[i];
      finalPts[i].grad.fill(0.0); // don't use gradient -- always fill with zeros

      if(globalPtWeights[i] > 1e-6) 
	finalPts[i].value = globalPtValues[i] / finalPts[i].weight;
      else 
	finalPts[i].value = 0.0; // no weight, so just set value to zero
    }
  }

  return;
}
#endif


void
QCAD::SaddleValueResponseFunction::
getImagePointValuesT(const double current_time,
		    const Tpetra_Vector* xdotT,
		    const Tpetra_Vector& xT,
		    const Teuchos::Array<ParamVec>& p,
		    Tpetra_Vector& gT, 
		    double* globalPtValues,
		    double* globalPtWeights,
		    double* globalPtGrads,
		    std::vector<QCAD::mathVector> lastPositions,
		    int dbMode)
{
  Teuchos::RCP<const Teuchos::Comm<int> > commT = xT.getMap()->getComm();

  //Set xmax,xmin,ymax,ymin based on points
  xmax = xmin = imagePts[0].coords[0];
  ymax = ymin = imagePts[0].coords[1];
  for(std::size_t i=1; i<nImagePts; i++) {
    xmin = std::min(imagePts[i].coords[0],xmin);
    xmax = std::max(imagePts[i].coords[0],xmax);
    ymin = std::min(imagePts[i].coords[1],ymin);
    ymax = std::max(imagePts[i].coords[1],ymax);
  }
  xmin -= 5*imagePtSize; xmax += 5*imagePtSize;
  ymin -= 5*imagePtSize; ymax += 5*imagePtSize;
    
  //Reset value, weight, and gradient of image points as these are accumulated by evaluator fill
  imagePtValues.fill(0.0);
  imagePtWeights.fill(0.0);
  imagePtGradComps.fill(0.0);

  if(bAggregateWorksets) {
    //Use cached field and coordinate values to perform fill    
    for(std::size_t i=0; i<vFieldValues.size(); i++) {
      addImagePointData( vCoords[i].data, vFieldValues[i], vGrads[i].data );
    } 
  }
  else {
    mode = "Collect image point data";
    Albany::FieldManagerScalarResponseFunction::evaluateResponseT(
				     current_time, xdotT, NULL, xT, p, gT);
  }

  //MPI -- sum weights, value, and gradient for each image pt
  Teuchos::reduceAll<LO, ST>(*commT, Teuchos::REDUCE_SUM, nImagePts, imagePtValues.data(), globalPtValues); 
  Teuchos::reduceAll<LO, ST>(*commT, Teuchos::REDUCE_SUM, nImagePts, imagePtWeights.data(), globalPtWeights); 
  Teuchos::reduceAll<LO, ST>(*commT, Teuchos::REDUCE_SUM, nImagePts*numDims, imagePtGradComps.data(), globalPtGrads); 
  //comm.SumAll( imagePtValues.data(),    globalPtValues,  nImagePts );
  //comm.SumAll( imagePtWeights.data(),   globalPtWeights, nImagePts );
  //comm.SumAll( imagePtGradComps.data(), globalPtGrads,   nImagePts*numDims );

  // Put summed data into imagePts, normalizing value and 
  //   gradient from different cell contributions
  for(std::size_t i=0; i<nImagePts; i++) {
    imagePts[i].weight = globalPtWeights[i];
    if(globalPtWeights[i] > 1e-6) {
      imagePts[i].value = globalPtValues[i] / imagePts[i].weight;
      for(std::size_t k=0; k<numDims; k++) 
	imagePts[i].grad[k] = globalPtGrads[k*nImagePts+i] / imagePts[i].weight;
    }
    else { 
      //assume point has drifted off region: leave value as, set gradient to zero, and reset position
      imagePts[i].grad.fill(0.0);
      imagePts[i].coords = lastPositions[i];
    }
  }

  return;
}
void
QCAD::SaddleValueResponseFunction::
writeOutput(int nIters)
{
  // Write output every nEvery iterations
  if( (nEvery > 0) && (nIters % nEvery == 1) && (outputFilename.length() > 0)) {
    std::fstream out; double pathLength = 0.0;
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << std::endl << std::endl << "% NEB Iteration " << nIters << std::endl;
    out << "% index xCoord yCoord value pathLength pointRadius" << std::endl;
    for(std::size_t i=0; i<nImagePts; i++) {
      out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	  << " " << imagePts[i].value << " " << pathLength << " " << imagePts[i].radius << std::endl;
      if(i<nImagePts-1) pathLength += imagePts[i].coords.distanceTo(imagePts[i+1].coords);
    }    
    out.close();
  }
}

void
QCAD::SaddleValueResponseFunction::
initialIterationSetup(double& gradScale, double& springScale, int dbMode)
{
  const QCAD::mathVector& initialPt = imagePts[0].coords;
  const QCAD::mathVector& finalPt = imagePts[nImagePts-1].coords;

  double maxGradMag = 0.0, avgWeight = 0.0, ifDist;
  for(std::size_t i=0; i<nImagePts; i++) {
    maxGradMag = std::max(maxGradMag, imagePts[i].grad.norm());
    avgWeight += imagePts[i].weight;
  }
  ifDist = initialPt.distanceTo(finalPt);
  avgWeight /= nImagePts;
  gradScale = ifDist / maxGradMag;  // want scale*maxGradMag*(dt=1.0) = distance btw initial & final pts

  // want springScale * (baseSpringConst=1.0) * (initial distance btwn pts) = scale*maxGradMag = distance btwn initial & final pts
  //  so springScale = (nImagePts-1)
  springScale = (double) (nImagePts-1);

  if(dbMode) std::cout << "Saddle Point:  First iteration:  maxGradMag=" << maxGradMag
		       << " |init-final|=" << ifDist << " gradScale=" << gradScale 
		       << " springScale=" << springScale << " avgWeight=" << avgWeight << std::endl;
  return;
}

void
QCAD::SaddleValueResponseFunction::
computeTangent(std::size_t i, QCAD::mathVector& tangent, int dbMode)
{
  // Compute tangent vector: use only higher neighboring imagePt.  
  //   Linear combination if both neighbors are above/below to smoothly interpolate cases
  double dValuePrev = imagePts[i-1].value - imagePts[i].value;
  double dValueNext = imagePts[i+1].value - imagePts[i].value;

  if(dValuePrev * dValueNext < 0.0) { //if both neighbors are either above or below current pt
    double dmax = std::max( fabs(dValuePrev), fabs(dValueNext) );
    double dmin = std::min( fabs(dValuePrev), fabs(dValueNext) );
    if(imagePts[i-1].value > imagePts[i+1].value)
      tangent = (imagePts[i+1].coords - imagePts[i].coords) * dmin + (imagePts[i].coords - imagePts[i-1].coords) * dmax;
    else
      tangent = (imagePts[i+1].coords - imagePts[i].coords) * dmax + (imagePts[i].coords - imagePts[i-1].coords) * dmin;
  }
  else {  //if one neighbor is above, the other below, just use the higher neighbor
    if(imagePts[i+1].value > imagePts[i].value)
      tangent = (imagePts[i+1].coords - imagePts[i].coords);
    else
      tangent = (imagePts[i].coords - imagePts[i-1].coords);
  }
  tangent.normalize();
  return;
}

void
QCAD::SaddleValueResponseFunction::
computeClimbingForce(std::size_t i, const QCAD::mathVector& tangent, const double& gradScale,
		     QCAD::mathVector& force, int dbMode)
{
  // Special case for highest point in climbing-NEB: force has full -Grad(V) but with parallel 
  //    component reversed and no force from springs (Future: add some perp spring force to avoid plateaus?)
  double dp = imagePts[i].grad.dot( tangent );
  force = (imagePts[i].grad * -1.0 + (tangent*dp) * 2) * gradScale; // force += -Grad(V) + 2*Grad(V)_parallel

  if(dbMode > 2) {
    std::cout << "Saddle Point:  --   tangent = " << tangent << std::endl;
    std::cout << "Saddle Point:  --   grad along tangent = " << dp << std::endl;
    std::cout << "Saddle Point:  --   total force (climbing) = " << force[i] << std::endl;
  }
}

void
QCAD::SaddleValueResponseFunction::
computeForce(std::size_t i, const QCAD::mathVector& tangent, const std::vector<double>& springConstants,
	     const double& gradScale,  const double& springScale, QCAD::mathVector& force,
	     double& dt, double& dt2, int dbMode)
{
  force.fill(0.0);
	
  // Get gradient projected perpendicular to the tangent and add to the force
  double dp = imagePts[i].grad.dot( tangent );
  force -= (imagePts[i].grad - tangent * dp) * gradScale; // force += -Grad(V)_perp

  if(dbMode > 2) {
    std::cout << "Saddle Point:  --   tangent = " << tangent << std::endl;
    std::cout << "Saddle Point:  --   grad along tangent = " << dp << std::endl;
    std::cout << "Saddle Point:  --   grad force = " << force[i] << std::endl;
  }

  // Get spring force projected parallel to the tangent and add to the force
  mathVector dNext(numDims), dPrev(numDims);
  mathVector parallelSpringForce(numDims), perpSpringForce(numDims);
  mathVector springForce(numDims);

  dPrev = imagePts[i-1].coords - imagePts[i].coords;
  dNext = imagePts[i+1].coords - imagePts[i].coords;

  double prevNorm = dPrev.norm();
  double nextNorm = dNext.norm();

  double perpFactor = 0.5 * (1 + cos(3.141592 * fabs(dPrev.dot(dNext) / (prevNorm * nextNorm))));
  springForce = ((dNext * springConstants[i]) + (dPrev * springConstants[i-1]));
  parallelSpringForce = tangent * springForce.dot(tangent);
  perpSpringForce = (springForce - tangent * springForce.dot(tangent) );
	
  springForce = parallelSpringForce + perpSpringForce * (perpFactor * antiKinkFactor);  
  while(springForce.norm() * dt2 > std::max(dPrev.norm(),dNext.norm()) && dt > minTimeStep) {
    dt /= 2; dt2=dt*dt;
    if(dbMode > 2) std::cout << "Saddle Point:  ** Warning: spring forces seem too large: dt => " << dt << std::endl;
  }

  force += springForce; // force += springForce_parallel + part of springForce_perp

  if(dbMode > 2) {
    std::cout << "Saddle Point:  --   spring force = " << springForce << std::endl;
    std::cout << "Saddle Point:  --   total force = " << force[i] << std::endl;
  }
}



std::string QCAD::SaddleValueResponseFunction::getMode()
{
  return mode;
}


bool QCAD::SaddleValueResponseFunction::
pointIsInImagePtRegion(const double* p, double refZ) const
{
  //assumes at least 2 dimensions
  if(numDims > 2 && (refZ < zmin || refZ > zmax)) return false;
  return !(p[0] < xmin || p[1] < ymin || p[0] > xmax || p[1] > ymax);
}

bool QCAD::SaddleValueResponseFunction::
pointIsInAccumRegion(const double* p, double refZ) const
{
  //assumes at least 2 dimensions
  if(numDims > 2 && (refZ < zmin || refZ > zmax)) return false;
  return true;
}

bool QCAD::SaddleValueResponseFunction::
pointIsInLevelSetRegion(const double* p, double refZ) const
{
  //assumes at least 2 dimensions
  if(numDims > 2 && (refZ < zmin || refZ > zmax)) return false;
  return (imagePts[iSaddlePt].coords.distanceTo(p) < levelSetRadius);
}



void QCAD::SaddleValueResponseFunction::
addBeginPointData(const std::string& elementBlock, const double* p, double value)
{
  //"Point" case: no need to process anything
  if( beginRegionType == "Point" ) return;

  //"Element Block" case: keep track of point with minimum value
  if( beginRegionType == "Element Block" ) {
    if(elementBlock == beginElementBlock) {
      if( value < imagePts[0].value || imagePts[0].weight == 0 ) {
	imagePts[0].value = value;
	imagePts[0].weight = 1.0; //positive weight flags initialization
	imagePts[0].coords.fill(p);
      }
    }
    return;
  }

  //"Polygon" case: keep track of point with minimum value
  if( beginRegionType == "Polygon" ) {
    if( QCAD::ptInPolygon(beginPolygon, p) ) {
      if( value < imagePts[0].value || imagePts[0].weight == 0 ) {
	imagePts[0].value = value;
	imagePts[0].weight = 1.0; //positive weight flags initialization
	imagePts[0].coords.fill(p);
      }
    }
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Invalid region type: " << beginRegionType << " for begin pt" << std::endl); 
  return;
}

void QCAD::SaddleValueResponseFunction::
addEndPointData(const std::string& elementBlock, const double* p, double value)
{
  //"Point" case: no need to process anything
  if( endRegionType == "Point" ) return;

  //"Element Block" case: keep track of point with minimum value
  if( endRegionType == "Element Block" ) {
    if(elementBlock == endElementBlock) {
      if( value < imagePts[nImagePts-1].value || imagePts[nImagePts-1].weight == 0 ) {
	imagePts[nImagePts-1].value = value;
	imagePts[nImagePts-1].weight = 1.0; //positive weight flags initialization
	imagePts[nImagePts-1].coords.fill(p);
      }
    }
    return;
  }

  //"Polygon" case: keep track of point with minimum value
  if( endRegionType == "Polygon" ) {
    if( QCAD::ptInPolygon(endPolygon, p) ) {
      if( value < imagePts[nImagePts-1].value || imagePts[nImagePts-1].weight == 0 ) {
	imagePts[nImagePts-1].value = value;
	imagePts[nImagePts-1].weight = 1.0; //positive weight flags initialization
	imagePts[nImagePts-1].coords.fill(p);
      }
    }
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Invalid region type: " << endRegionType << " for end pt" << std::endl); 
  return;
}


void QCAD::SaddleValueResponseFunction::
addImagePointData(const double* p, double value, double* grad)
{
  double w, effDims = (bLockToPlane && numDims > 2) ? 2 : numDims;
  for(std::size_t i=0; i<nImagePts; i++) {
    w = pointFn(imagePts[i].coords.distanceTo(p) , imagePts[i].radius );
    if(w > 0) {
      imagePtWeights[i] += w;
      imagePtValues[i] += w*value;
      for(std::size_t k=0; k<effDims; k++)
	imagePtGradComps[k*nImagePts+i] += w*grad[k];
      //std::cout << "DEBUG Image Pt " << i << " close to (" << p[0] << "," << p[1] << "," << p[2] << ")=" << value
      //	<< "  wt=" << w << "  totalW=" << imagePtWeights[i] << "  totalVal=" << imagePtValues[i]
      //	<< "  val=" << imagePtValues[i] / imagePtWeights[i] << std::endl;
    }
  }
  return;
}

void QCAD::SaddleValueResponseFunction::
addFinalImagePointData(const double* p, double value)
{
  double w;
  for(std::size_t i=0; i< finalPts.size(); i++) {
    w = pointFn(finalPts[i].coords.distanceTo(p) , finalPts[i].radius );
    if(w > 0) {
      finalPtWeights[i] += w;
      finalPtValues[i] += w*value;
    }
  }
  return;
}

void QCAD::SaddleValueResponseFunction::
accumulatePointData(const double* p, double value, double* grad)
{
  vFieldValues.push_back(value);
  vCoords.push_back( QCAD::maxDimPt(p,numDims) );
  vGrads.push_back( QCAD::maxDimPt(grad,numDims) );
}


void QCAD::SaddleValueResponseFunction::
accumulateLevelSetData(const double* p, double value, double cellArea)
{
  vlsFieldValues.push_back(value);
  vlsCellAreas.push_back(cellArea);
  for(std::size_t i=0; i < numDims; ++i)
    vlsCoords[i].push_back(p[i]);
}
 

//Adds and returns the weight of a point relative to the saddle point position.
double QCAD::SaddleValueResponseFunction::
getSaddlePointWeight(const double* p) const
{
  return pointFn(imagePts[iSaddlePt].coords.distanceTo(p) , imagePts[iSaddlePt].radius );
}

double QCAD::SaddleValueResponseFunction::
getTotalSaddlePointWeight() const
{
  return imagePts[iSaddlePt].weight;
}

const double* QCAD::SaddleValueResponseFunction::
getSaddlePointPosition() const
{
  return imagePts[iSaddlePt].coords.data();
}

#if defined(ALBANY_EPETRA)
bool QCAD::SaddleValueResponseFunction::
matchesCurrentResults(Epetra_Vector& g) const
{
  const double TOL = 1e-8;

  if(iSaddlePt < 0) return false;

  if( fabs(g[0] - returnFieldVal) > TOL || fabs(g[1] - imagePts[iSaddlePt].value) > TOL)
    return false;

  for(std::size_t i=0; i<numDims; i++) {
    if(  fabs(g[2+i] - imagePts[iSaddlePt].coords[i]) > TOL ) return false;
  }

  return true;
}
#endif

bool QCAD::SaddleValueResponseFunction::
matchesCurrentResultsT(Tpetra_Vector& gT) const
{
  const double TOL = 1e-8;

  Teuchos::ArrayRCP<const ST> gT_constView = gT.get1dView();

  if(iSaddlePt < 0) return false;

  if( fabs(gT_constView[0] - returnFieldVal) > TOL || fabs(gT_constView[1] - imagePts[iSaddlePt].value) > TOL)
    return false;

  for(std::size_t i=0; i<numDims; i++) {
    if(  fabs(gT_constView[2+i] - imagePts[iSaddlePt].coords[i]) > TOL ) return false;
  }

  return true;
}


double QCAD::SaddleValueResponseFunction::
pointFn(double d, double radius) const {
  //return ( d < radius ) ? 1.0 : 0.0;  //alternative?

  const double N = 1.0;
  double val = N*exp(-d*d / (2*radius*radius));
  return (val >= 1e-2) ? val : 0.0;
}

int QCAD::SaddleValueResponseFunction::
getHighestPtIndex() const 
{
  // Find the highest image point 
  int iHighestPt = 0; // init to the first point being the highest
  for(std::size_t i=1; i<nImagePts; i++) {
    if(imagePts[i].value > imagePts[iHighestPt].value) iHighestPt = i;
  }
  return iHighestPt;
}



/*************************************************************/
//! Helper functions
/*************************************************************/

/*double distanceFromLineSegment(const double* p, const double* p1, const double* p2, int dims)
{
  //get pt c, the point along the full line p1->p2 closest to p
  double s, dp = 0, mag2 = 0; 

  for(int k=0; k<dims; k++) mag2 += pow(p2[k]-p1[k],2);
  for(int k=0; k<dims; k++) dp += (p[k]-p1[k])*(p2[k]-p1[k]);
  s = dp / sqrt(mag2); // < 0 or > 1 if c is outside segment, 0 <= s <= 1 if c is on segment

  if(0 <= s && s <= 1) { //just return distance between c and p
    double cp = 0;
    for(int k=0; k<dims; k++) cp += pow(p[k] - (p1[k]+s*(p2[k]-p1[k])),2);
    return sqrt(cp);
  }
  else { //take closer distance from the endpoints
    double d1=0, d2=0;
    for(int k=0; k<dims; k++) d1 += pow(p[k]-p1[k],2);
    for(int k=0; k<dims; k++) d2 += pow(p[k]-p2[k],2);
    if(d1 < d2) return sqrt(d1);
    else return sqrt(d2);
  }
  }*/


//MOVED to QCAD_MathVector.cpp
/*
// Returns true if point is inside polygon, false otherwise
//  Assumes 2D points (more dims ok, but uses only first two components)
//  Algorithm = ray trace along positive x-axis
bool QCAD::ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const QCAD::mathVector& pt) 
{
  return QCAD::ptInPolygon(polygon, pt.data());
}

bool QCAD::ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const double* pt)
{
  bool c = false;
  int n = (int)polygon.size();
  double x=pt[0], y=pt[1];
  const int X=0,Y=1;

  for (int i = 0, j = n-1; i < n; j = i++) {
    const QCAD::mathVector& pi = polygon[i];
    const QCAD::mathVector& pj = polygon[j];
    if ((((pi[Y] <= y) && (y < pj[Y])) ||
	 ((pj[Y] <= y) && (y < pi[Y]))) &&
	(x < (pj[X] - pi[X]) * (y - pi[Y]) / (pj[Y] - pi[Y]) + pi[X]))
      c = !c;
  }
  return c;
}
*/

//Not used - but keep for reference
// Returns true if point is inside polygon, false otherwise
/*bool orig_ptInPolygon(int npol, float *xp, float *yp, float x, float y)
{
  int i, j; bool c = false;
  for (i = 0, j = npol-1; i < npol; j = i++) {
    if ((((yp[i] <= y) && (y < yp[j])) ||
	 ((yp[j] <= y) && (y < yp[i]))) &&
	(x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
      c = !c;
  }
  return c;
}*/






/*************************************************************/
//! mathVector class: a vector with math operations (helper class) 
/*************************************************************/

//MOVED to it's own file (QCAD_MathVector.cpp)
/*
QCAD::mathVector::mathVector() 
{
  dim_ = -1;
}

QCAD::mathVector::mathVector(int n) 
 :data_(n) 
{ 
  dim_ = n;
}


QCAD::mathVector::mathVector(const mathVector& copy) 
{ 
  data_ = copy.data_;
  dim_ = copy.dim_;
}

QCAD::mathVector::~mathVector() 
{
}


void 
QCAD::mathVector::resize(std::size_t n) 
{ 
  data_.resize(n); 
  dim_ = n;
}

void 
QCAD::mathVector::fill(double d) 
{ 
  for(int i=0; i<dim_; i++) data_[i] = d;
}

void 
QCAD::mathVector::fill(const double* vec) 
{ 
  for(int i=0; i<dim_; i++) data_[i] = vec[i];
}

double 
QCAD::mathVector::dot(const mathVector& v2) const
{
  double d=0;
  for(int i=0; i<dim_; i++)
    d += data_[i] * v2[i];
  return d;
}

QCAD::mathVector& 
QCAD::mathVector::operator=(const mathVector& rhs)
{
  data_ = rhs.data_;
  dim_ = rhs.dim_;
  return *this;
}

QCAD::mathVector 
QCAD::mathVector::operator+(const mathVector& v2) const
{
  mathVector result(dim_);
  for(int i=0; i<dim_; i++) result[i] = data_[i] + v2[i];
  return result;
}

QCAD::mathVector 
QCAD::mathVector::operator-(const mathVector& v2) const
{
  mathVector result(dim_);
  for(int i=0; i<dim_; i++) result[i] = data_[i] - v2[i];
  return result;
}

QCAD::mathVector
QCAD::mathVector::operator*(double scale) const 
{
  mathVector result(dim_);
  for(int i=0; i<dim_; i++) result[i] = scale*data_[i];
  return result;
}

QCAD::mathVector& 
QCAD::mathVector::operator+=(const mathVector& v2)
{
  for(int i=0; i<dim_; i++) data_[i] += v2[i];
  return *this;
}

QCAD::mathVector& 
QCAD::mathVector::operator-=(const mathVector& v2) 
{
  for(int i=0; i<dim_; i++) data_[i] -= v2[i];
  return *this;
}

QCAD::mathVector& 
QCAD::mathVector::operator*=(double scale) 
{
  for(int i=0; i<dim_; i++) data_[i] *= scale;
  return *this;
}

QCAD::mathVector&
QCAD::mathVector::operator/=(double scale) 
{
  for(int i=0; i<dim_; i++) data_[i] /= scale;
  return *this;
}

double&
QCAD::mathVector::operator[](int i) 
{ 
  return data_[i];
}

const double& 
QCAD::mathVector::operator[](int i) const 
{ 
  return data_[i];
}

double 
QCAD::mathVector::distanceTo(const mathVector& v2) const
{
  double d = 0.0;
  for(int i=0; i<dim_; i++) d += pow(data_[i]-v2[i],2);
  return sqrt(d);
}

double 
QCAD::mathVector::distanceTo(const double* p) const
{
  double d = 0.0;
  for(int i=0; i<dim_; i++) d += pow(data_[i]-p[i],2);
  return sqrt(d);
}
				 
double 
QCAD::mathVector::norm() const
{ 
  return sqrt(dot(*this)); 
}

double 
QCAD::mathVector::norm2() const
{ 
  return dot(*this); 
}


void 
QCAD::mathVector::normalize() 
{
  (*this) /= norm(); 
}

const double* 
QCAD::mathVector::data() const
{ 
  return data_.data();
}

double* 
QCAD::mathVector::data()
{ 
  return data_.data();
}


std::size_t 
QCAD::mathVector::size() const
{
  return dim_; 
}

std::ostream& QCAD::operator<<(std::ostream& os, const QCAD::mathVector& mv) 
{
  std::size_t size = mv.size();
  os << "(";
  for(std::size_t i=0; i<size-1; i++) os << mv[i] << ", ";
  if(size > 0) os << mv[size-1];
  os << ")";
  return os;
}
*/

std::ostream& QCAD::operator<<(std::ostream& os, const QCAD::nebImagePt& np)
{
  os << std::endl;
  os << "coords = " << np.coords << std::endl;
  os << "veloc  = " << np.velocity << std::endl;
  os << "grad   = " << np.grad << std::endl;
  os << "value  = " << np.value << std::endl;
  os << "weight = " << np.weight << std::endl;
  return os;
}



/*************************************************************/
//! Helper functions
/*************************************************************/

#if defined(ALBANY_EPETRA)
void QCAD::gatherVector(std::vector<double>& v, std::vector<double>& gv, const Epetra_Comm& comm_)
{
  double *pvec, zeroSizeDummy = 0;
  pvec = (v.size() > 0) ? &v[0] : &zeroSizeDummy;

  Epetra_Map map(-1, v.size(), 0, comm_);
  Epetra_Vector ev(View, map, pvec);
  int  N = map.NumGlobalElements();
  Epetra_LocalMap lomap(N,0,comm_);

  gv.resize(N);
  pvec = (gv.size() > 0) ? &gv[0] : &zeroSizeDummy;
  Epetra_Vector egv(View, lomap, pvec);
  Epetra_Import import(lomap,map);
  egv.Import(ev, import, Insert);
}
#endif

void QCAD::gatherVectorT(std::vector<double>& v, std::vector<double>& gv, Teuchos::RCP<const Teuchos::Comm<int> >& commT)
{
  double *pvec, zeroSizeDummy = 0;
  pvec = (v.size() > 0) ? &v[0] : &zeroSizeDummy;

  Tpetra::global_size_t numGlobalElements = Teuchos::OrdinalTraits<size_t>::invalid(); 
  Tpetra::LocalGlobal lg = Tpetra::GloballyDistributed;
  Teuchos::RCP<Tpetra_Map> mapT = Teuchos::rcp(new Tpetra_Map(numGlobalElements, 0, commT, lg));
  Teuchos::ArrayView<ST> pvecView = Teuchos::arrayView(pvec, v.size());  
  Teuchos::RCP<Tpetra_Vector> evT = Teuchos::rcp(new Tpetra_Vector(mapT, pvecView)); 
  int  N = mapT->getGlobalNumElements();
  lg = Tpetra::LocallyReplicated;
  Teuchos::RCP<Tpetra_Map> lomapT = Teuchos::rcp(new Tpetra_Map(N, 0, commT, lg)); //local map

  gv.resize(N);
  pvec = (gv.size() > 0) ? &gv[0] : &zeroSizeDummy;
  Teuchos::RCP<Tpetra_Vector> egvT = Teuchos::rcp(new Tpetra_Vector(lomapT, pvecView)); 
  Teuchos::RCP<Tpetra_Import> importT = Teuchos::rcp(new Tpetra_Import(mapT,lomapT));
  egvT->doImport(*evT, *importT, Tpetra::INSERT);
}

bool QCAD::lessOp(std::pair<std::size_t, double> const& a,
	    std::pair<std::size_t, double> const& b) {
  return a.second < b.second;
}

void QCAD::getOrdering(const std::vector<double>& v, std::vector<int>& ordering)
{
  typedef std::vector<double>::const_iterator dbl_iter;
  typedef std::vector<std::pair<std::size_t, double> >::const_iterator pair_iter;
  std::vector<std::pair<std::size_t, double> > vPairs(v.size());

  size_t n = 0;
  for (dbl_iter it = v.begin(); it != v.end(); ++it, ++n)
    vPairs[n] = std::make_pair(n, *it);


  std::sort(vPairs.begin(), vPairs.end(), QCAD::lessOp);

  ordering.resize(v.size()); n = 0;
  for (pair_iter it = vPairs.begin(); it != vPairs.end(); ++it, ++n)
    ordering[n] = it->first;
}


double QCAD::averageOfVector(const std::vector<double>& v)
{
  double avg = 0.0;
  for(std::size_t i=0; i < v.size(); i++) {
    avg += v[i];
  }
  avg /= v.size();
  return avg;
}

double QCAD::distance(const std::vector<double>* vCoords, int ind1, int ind2, std::size_t nDims)
{
  double d2 = 0;
  for(std::size_t k=0; k<nDims; k++)
    d2 += pow( vCoords[k][ind1] - vCoords[k][ind2], 2 );
  return sqrt(d2);
}
