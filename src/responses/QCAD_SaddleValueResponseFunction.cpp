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

#include <Teuchos_Array.hpp>
#include <Epetra_LocalMap.h>
#include "Albany_Utils.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"
#include <fstream>

//! Helper function prototypes
bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const QCAD::mathVector& pt);
bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const double* pt);


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

  imagePtSize   = params.get<double>("Image Point Size", 0.01);
  nImagePts     = params.get<int>("Number of Image Points", 10);
  maxTimeStep   = params.get<double>("Max Time Step", 1.0);
  minTimeStep   = params.get<double>("Min Time Step", 0.002);
  maxIterations = params.get<int>("Maximum Iterations", 100);
  convergeTolerance = params.get<double>("Convergence Tolerance", 1e-5);
  minSpringConstant = params.get<double>("Min Spring Constant", 1.0);
  maxSpringConstant = params.get<double>("Max Spring Constant", 1.0);
  outputFilename = params.get<std::string>("Output Filename", "");
  debugFilename  = params.get<std::string>("Debug Filename", "");
  nEvery         = params.get<int>("Output Interval", 0);
  bClimbing      = params.get<bool>("Climbing NEB", true);
  antiKinkFactor = params.get<double>("Anti-Kink Factor", 0.0);

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
  bPositiveOnly = params.get<bool>("Positive Return Only",false);

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
evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g)
{
  const Epetra_Comm& comm = x.Map().Comm();

  int dbMode = (comm.MyPID() == 0) ? debugMode : 0;
  if(comm.MyPID() != 0) outputFilename = ""; //Only root process outputs to files
  if(comm.MyPID() != 0) debugFilename = ""; //Only root process outputs to files
  

  TEUCHOS_TEST_FOR_EXCEPTION (nImagePts < 2, Teuchos::Exceptions::InvalidParameter, std::endl 
	      << "Saddle Point needs more than 2 image pts (" << nImagePts << " given)" << std::endl); 

  // Find saddle point in stages:

  // 1) Determine initial and final points depending on region type
  //     - Point: take point given directly
  //     - Element Block: take minimum point within the specified element block (and allowed z-range)
  //     - Polygon: take minimum point within specified 2D polygon and allowed z-range
  
  if(dbMode > 1) std::cout << "Saddle Point:  Beginning end point location" << std::endl;

    // Initialize intial/final points
  imagePts[0].init(numDims);
  imagePts[nImagePts-1].init(numDims);

  mode = "Point location";
  Albany::FieldManagerScalarResponseFunction::evaluateResponse(
	current_time, xdot, x, p, g);
  if(dbMode > 2) std::cout << "Saddle Point:   -- done evaluation" << std::endl;

  if(beginRegionType == "Point") {
    imagePts[0].coords = beginPolygon[0];
  }
  else { //MPI: get global min for begin point
    double globalMin; int procToBcast, winner;
    comm.MinAll( &imagePts[0].value, &globalMin, 1);
    if( fabs(imagePts[0].value - globalMin) < 1e-8 ) 
      procToBcast = comm.MyPID();
    else procToBcast = -1;

    comm.MaxAll( &procToBcast, &winner, 1 );
    comm.Broadcast( imagePts[0].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[0].value = globalMin;                               //no need to broadcast winner's value
  }

  if(endRegionType   == "Point") {
    imagePts[nImagePts-1].coords = endPolygon[0];
  }
  else { //MPI: get global min for end point
    double globalMin; int procToBcast, winner;
    comm.MinAll( &imagePts[nImagePts-1].value, &globalMin, 1);
    if( fabs(imagePts[nImagePts-1].value - globalMin) < 1e-8 ) 
      procToBcast = comm.MyPID();
    else procToBcast = -1;

    comm.MaxAll( &procToBcast, &winner, 1 );
    comm.Broadcast( imagePts[nImagePts-1].coords.data(), numDims, winner); //broadcast winner's min position to others
    imagePts[nImagePts-1].value = globalMin;                               //no need to broadcast winner's value
  }

  if(dbMode > 2) std::cout << "Saddle Point:   -- done begin/end point initialization" << std::endl;


  // Initialize Image Points:  interpolate between initial and final points (and possibly guess point) to get all the image points
  const mathVector& initialPt = imagePts[0].coords;
  const mathVector& finalPt   = imagePts[nImagePts-1].coords;
  if(saddleGuessGiven) {
    // two line segements (legs) initialPt -> guess, guess -> finalPt
    int nFirstLeg = (nImagePts+1)/2, nSecondLeg = nImagePts - nFirstLeg + 1; // +1 because both legs include middle pt
    for(int i=1; i<nFirstLeg-1; i++) {
      double s = i * 1.0/(nFirstLeg-1);
      imagePts[i].init(initialPt + (saddlePointGuess - initialPt) * s);
    }
    for(int i=0; i<nSecondLeg-1; i++) {
      double s = i * 1.0/(nSecondLeg-1);
      imagePts[i+nFirstLeg-1].init(saddlePointGuess + (finalPt - saddlePointGuess) * s);
    }
  }
  else {
    // one line segment initialPt -> finalPt
    for(std::size_t i=1; i<nImagePts-1; i++) {
      double s = i * 1.0/(nImagePts-1);   // nIntervals = nImagePts-1
      imagePts[i].init(initialPt + (finalPt - initialPt) * s);
    }     
  }
 
  if(dbMode > 1) {
    for(std::size_t i=0; i<nImagePts; i++)
      std::cout << "Saddle Point:   -- imagePt[" << i << "] = " << imagePts[i].coords << std::endl;
  }
  
  //  2) Perform Nudged Elastic Band Algorithm to find saddle point.
  //      Iterate over field manager fills of each image point's value and gradient
  //       then update image point positions user Verlet algorithm
  
  mode = "Collect image point data";
  std::size_t nIters, nInitialIterations, nDecreases=0;
  mathVector tangent(numDims), springForce(numDims), parallelSpringForce(numDims), perpSpringForce(numDims), dNext(numDims), dPrev(numDims);
  double mag, dp, maxUpdate, dValuePrev, dValueNext, lastHighestVal=0, avgForce=0, avgOpposingForce=0;
  int nConsecHighForceDiff=0, nConsecLowForceDiff=0;
  std::vector<mathVector> force(nImagePts), lastForce(nImagePts), lastPositions(nImagePts);
  std::vector<double> springConstants(nImagePts-1, minSpringConstant);
  for(std::size_t i=0; i<nImagePts; i++) force[i].resize(numDims);

  double dt = maxTimeStep;
  double dt2 = dt*dt;
  double gradScale   = 1.0;
  double springScale = 1.0;
  double springBase = 1.0;
  nIters = 0;
  nInitialIterations = 20;
  bool decreaseMode = true; // start by waiting until highest point does not decrease in value
  
  //Allocate storage for aggrecated image point data (needed for MPI)
  double*  globalPtValues   = new double [nImagePts];
  double*  globalPtWeights  = new double [nImagePts];
  double*  globalPtGrads    = new double [nImagePts*numDims];

  //Write initial header to output file
  if( outputFilename.length() > 0) {
    std::fstream out;
    out.open(outputFilename.c_str(), std::fstream::out);
    out << "# Saddle point path" << std::endl;
    out << "# index xCoord yCoord value" << std::endl;
    out.close();
  }

  std::fstream fDebug;
  if(debugFilename.length() > 0) {
    fDebug.open(debugFilename.c_str(), std::fstream::out);
    fDebug << "# HighestValue  HighestIndex  AverageForce  TimeStep"
	   << "  HighestPtGradNorm  AverageOpposingForce  SpringBase" << std::endl;
  }
  
  while( ++nIters <= maxIterations) {

    // Write output every nEvery iterations
    if( (nEvery > 0) && (nIters % nEvery == 1) && (outputFilename.length() > 0)) {
      std::fstream out;
      out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
      out << "# Iteration " << nIters << std::endl;
      for(std::size_t i=0; i<nImagePts; i++) {
	out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	    << " " << imagePts[i].value << std::endl;
      }    
      out << std::endl << std::endl; //dataset separation
      out.close();
    }
    
    if(dbMode > 1) std::cout << "Saddle Point:  NEB Algorithm iteration " << nIters << " -----------------------" << std::endl;
    
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

    Albany::FieldManagerScalarResponseFunction::evaluateResponse(
				       current_time, xdot, x, p, g);

    //MPI -- sum weights, value, and gradient for each image pt
    comm.SumAll( imagePtValues.data(),    globalPtValues,  nImagePts );
    comm.SumAll( imagePtWeights.data(),   globalPtWeights, nImagePts );
    comm.SumAll( imagePtGradComps.data(), globalPtGrads,   nImagePts*numDims );

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
      
      //on first iteration adjust gradient scaling
    if(nIters == 1) { 
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

      if(dbMode > 2) std::cout << "Saddle Point:  First iteration:  maxGradMag=" << maxGradMag
			       << " |init-final|=" << ifDist << " gradScale=" << gradScale 
			       << " springScale=" << springScale << " avgWeight=" << avgWeight << std::endl;

      for(std::size_t i=1; i<nImagePts-1; i++) lastForce[i] = force[i]; //initialize last force
    }

    // Compute spring base constant for this run
    double s = ((double)nIters-1.0)/maxIterations;    
    springBase = springScale * ( (1.0-s)*minSpringConstant + s*maxSpringConstant ); 
    for(std::size_t i=0; i<nImagePts-1; i++) springConstants[i] = springBase;
	  
      
    // Find highest point if using the climbing NEB technique
    int iHighestPt = 0; //effectively a null since loops below begin at 1
    for(std::size_t i=1; i<nImagePts-1; i++) {
      if(imagePts[i].value > imagePts[iHighestPt].value) iHighestPt = i;
    }

    // Compute tangents between points and project gradient and elastic forces appropriately
    for(std::size_t i=1; i<nImagePts-1; i++) {
      
      if(dbMode > 2) std::cout << std::endl << "Saddle Point:  >> Updating pt[" << i << "]:" << imagePts[i];

      // Compute tangent vector: use only higher neighboring imagePt.  
      //   Linear combination if both neighbors are above/below to smoothly interpolate cases
      dValuePrev = imagePts[i-1].value - imagePts[i].value;
      dValueNext = imagePts[i+1].value - imagePts[i].value;
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
	

      if((int)i == iHighestPt && bClimbing && nIters > nInitialIterations) {
	// Special case for highest point in climbing-NEB: force has full -Grad(V) but with parallel 
	//    component reversed and no force from springs
	dp = imagePts[i].grad.dot( tangent );
	force[i] = (imagePts[i].grad * -1.0 + (tangent*dp) * 2) * gradScale; // force += -Grad(V) + 2*Grad(V)_parallel

	/*dPrev = imagePts[i-1].coords - imagePts[i].coords;
	dNext = imagePts[i+1].coords - imagePts[i].coords;
	double perpFactor = 0.5 * (1 + cos(3.141592 * fabs(dPrev.dot(dNext) / (dPrev.norm() * dNext.norm()))));
	springForce = ((dNext * springConstants[i]) + (dPrev * springConstants[i-1]));
	dp = springForce.dot(tangent);
	force[i] += (springForce - tangent*dp) * perpFactor * antiKinkFactor; */

	if(dbMode > 2) {
	  std::cout << "Saddle Point:  --   tangent = " << tangent << std::endl;
	  std::cout << "Saddle Point:  --   grad along tangent = " << dp << std::endl;
	  std::cout << "Saddle Point:  --   total force (climbing) = " << force[i] << std::endl;
	}
      }
      else {
	force[i].fill(0.0);
	
	// Get gradient projected perpendicular to the tangent and add to the force
	dp = imagePts[i].grad.dot( tangent );
	force[i] -= (imagePts[i].grad - tangent * dp) * gradScale; // force += -Grad(V)_perp

	if(dbMode > 2) {
	  std::cout << "Saddle Point:  --   tangent = " << tangent << std::endl;
	  std::cout << "Saddle Point:  --   grad along tangent = " << dp << std::endl;
	  std::cout << "Saddle Point:  --   grad force = " << force[i] << std::endl;
	}

	// Get spring force projected parallel to the tangent and add to the force
	dPrev = imagePts[i-1].coords - imagePts[i].coords;
	dNext = imagePts[i+1].coords - imagePts[i].coords;

	double perpFactor = 0.5 * (1 + cos(3.141592 * fabs(dPrev.dot(dNext) / (dPrev.norm() * dNext.norm()))));
	springForce = ((dNext * springConstants[i]) + (dPrev * springConstants[i-1]));
	parallelSpringForce = tangent * springForce.dot(tangent);
	//parallelSpringForce = tangent * ((dNext.norm() * springConstants[i]) - (dPrev.norm() * springConstants[i-1]));
	//  Note: this expression can be problematic when distance to lower neighbor is much larger than higher neighbor (tangent dir)
	perpSpringForce = (springForce - tangent * springForce.dot(tangent) );
	
	springForce = parallelSpringForce + perpSpringForce * (perpFactor * antiKinkFactor);  
	while(springForce.norm() * dt2 > std::max(dPrev.norm(),dNext.norm()) && dt > minTimeStep) {
	  dt /= 2; dt2=dt*dt;
	  if(dbMode > 2) std::cout << "Saddle Point:  ** Warning: spring forces seem too large: dt => " << dt << std::endl;
	}

	force[i] += springForce; // force += springForce_parallel + part of springForce_perp

	if(dbMode > 2) {
	  std::cout << "Saddle Point:  --   spring force = " << springForce << std::endl;
	  std::cout << "Saddle Point:  --   total force = " << force[i] << std::endl;
	}
      }

      if(dbMode > 2) std::cout << "Saddle Point:  --   Force on pt[" << i << "] = " << imagePts[i].value 
				  << " : " << imagePts[i].coords << " is " << force[i] << std::endl;
    }
      
    if(dbMode > 1) std::cout << "Saddle Point:  ** Summary:"
			     << "  highest val[" << iHighestPt << "] = " << imagePts[iHighestPt].value
			     << "  AverageForce = " << avgForce << "  dt = " << dt 
			     << "  gradNorm = " << imagePts[iHighestPt].grad.norm() 
			     << "  AvgOpposingForce = " << avgOpposingForce 
			     << "  SpringBase = " << springBase << std::endl;
    
    if(debugFilename.length() > 0) {
      fDebug << imagePts[iHighestPt].value << "  "
	     << iHighestPt << "  "
	     << avgForce << "  "
	     << dt << "  "
	     << imagePts[iHighestPt].grad.norm() << "  "
	     << avgOpposingForce << "  "
	     << springBase << std::endl;
    }

    if(imagePts[iHighestPt].grad.norm() < convergeTolerance) {
      if(dbMode > 2) std::cout << "Saddle Point:  ** Converged (grad norm " << 
	       imagePts[iHighestPt].grad.norm() << " < " << convergeTolerance << ")" << std::endl;
      break; // exit iterations loop
    }

    // Check for convergence and modify dt if necessary
    /*if(decreaseMode == true) { // modify dt to get to point where even a small dt results in the highest point increasing in value
      if(nIters > nInitialIterations) {
	if(imagePts[iHighestPt].value <= lastHighestVal ) nDecreases++;
	else { // increase in highest point value => decrease dt
	  dt /= 2; dt2 = dt*dt;
	  if(dbMode > 2) std::cout << "Saddle Point:  ** Increase in max-point => dt = " << dt << std::endl;
	  if(dt < minTimeStep) {
	    if(dbMode > 2) std::cout << "Saddle Point:  ** dt below min time step => decrease mode done!" << std::endl;
	    decreaseMode = false;
	  }
	  nDecreases = 0;
	}
	
	if(nDecreases >= 10) {
	  if( dt * 2 < maxTimeStep ) {
	    dt *= 2; dt2 = dt*dt;
	  }
	  if(dbMode > 2) std::cout << "Saddle Point:  ** 10 consecutive decreases => dt = " << dt << std::endl;
	  nDecreases = 0;
	}
      }
      }*/
	  
    // in the case of climbing NEB, we want to allow the highest point to increase (b/c it climbs)
    /*if(!decreaseMode) {
      dt = minTimeStep; dt2 = dt*dt;
      if(dbMode > 2) std::cout << "Saddle Point:  ** Climbing mode => dt = " << dt << std::endl;
      if(imagePts[iHighestPt].grad.norm() < gradTolerance) {
	if(dbMode > 2) std::cout << "Saddle Point:  ** Converged (grad norm < " << gradTolerance << ")" << std::endl;
	break; // exit loop: no decrease in highest point even using smallest time step
      }
    }
    lastHighestVal = imagePts[iHighestPt].value;*/
       
    avgOpposingForce = 0.0;
    for(std::size_t i=1; i<nImagePts-1; i++) {
      dp = force[i].dot(lastForce[i]); //see if current force and last force point in the same direction
      dp /= (force[i].norm() * lastForce[i].norm()); // dot product btwn unit vectors => angle
      if( dp < 0 ) {//avgOpposingForce += dp;
	mathVector v = force[i] - lastForce[i];
	avgOpposingForce += v.norm() / (force[i].norm() + lastForce[i].norm());
      }
    }
    avgOpposingForce /= (nImagePts-2);
    for(std::size_t i=1; i<nImagePts-1; i++) lastForce[i] = force[i];

    //if(avgOpposingForce > 0.1) nConsecHighForceDiff++;
    /*if(avgOpposingForce > std::max( 0.2*(1.0 - nIters / ((double)maxIterations)), 0.01)) nConsecHighForceDiff++;
    else nConsecHighForceDiff = 0;

    if(nConsecHighForceDiff >= 1 && dt > minTimeStep) { 
      dt /= 2; dt2=dt*dt;
      nConsecHighForceDiff = 0;
      if(dbMode > 2) std::cout << "Saddle Point:  ** Consecutive high dForce => dt = " << dt << std::endl;
      }*/

    //if(avgOpposingForce < std::max( 0.01*(1.0 - nIters / ((double)maxIterations)), 0.0)) nConsecLowForceDiff++;
    if(avgOpposingForce < 1e-6) nConsecLowForceDiff++;
    else nConsecLowForceDiff = 0;

    if(nConsecLowForceDiff >= 3 && dt < maxTimeStep) { 
      dt *= 2; dt2=dt*dt;
      nConsecLowForceDiff = 0;
      if(dbMode > 2) std::cout << "Saddle Point:  ** Consecutive low dForce => dt = " << dt << std::endl;
    }
	  
    
    avgForce = 0.0;

    //Update points using precomputed forces
    for(std::size_t i=1; i<nImagePts-1; i++) {

      //Update coordinates and velocity using (modified) Verlet integration

      //Reset velocity to zero if it is directed opposite to force (reduces overshoot)
      dp = imagePts[i].velocity.dot(force[i]);
      if(dp < 0) imagePts[i].velocity.fill(0.0);  

      mathVector dCoords = imagePts[i].velocity * dt + force[i] * dt2;
      lastPositions[i] = imagePts[i].coords; //save last position in case the new position brings us outside the mesh
      imagePts[i].coords += dCoords;
      imagePts[i].velocity += force[i] * dt;

      //weight average force by distance from highest pt
      avgForce += (1.0 - fabs((double)i-(double)iHighestPt)/((double)nImagePts)) * force[i].norm(); 
    }
    avgForce /= nImagePts;

    /*//Check for convergence
    if(avgForce < convergeTolerance) {
      if(dbMode > 2) std::cout << "Saddle Point:  ** Converged (avg force " << avgForce << " < " << convergeTolerance << ")" << std::endl;
      break; // exit loop: no decrease in highest point even using smallest time step
    }*/
  }

  //deallocate storage used for MPI communication
  delete [] globalPtValues; 
  delete [] globalPtWeights;
  delete [] globalPtGrads;  

  // Check if converged: nIters < maxIters ?
  if(dbMode) std::cout << "Saddle Point:  Done NEB after " << nIters << " iterations" << std::endl;

  if(dbMode) {
    for(std::size_t i=0; i<nImagePts; i++) {
      std::cout << "Saddle Point:  --   Final pt[" << i << "] = " << imagePts[i].value 
		<< " : " << imagePts[i].coords << std::endl;
    }
  }

   
  // Choose image point with highest value as saddle point
  std::size_t imax = 0;
  for(std::size_t i=1; i<nImagePts; i++) {
    if(imagePts[i].value > imagePts[imax].value) imax = i;
  }
  saddlePt = imagePts[imax].coords;
  
  if(dbMode > 1) std::cout << "Saddle Point:  Begin filling saddle point data" << std::endl;
  mode = "Fill saddle point";
  Albany::FieldManagerScalarResponseFunction::evaluateResponse(
				   current_time, xdot, x, p, g);
  //MPI: saddle weight is already summed in evaluator's postEvaluate, so no need to do anything here

  if(dbMode > 1) std::cout << "Saddle Point:  Done filling saddle point data" << std::endl;

  // Overwrite response indices 2+ with saddle point coordinates
  for(std::size_t i=0; i<numDims; i++) g[2+i] = saddlePt[i]; 

  if(dbMode) {
    std::cout << "Saddle Point:  Return Field value = " << g[0] << std::endl;
    std::cout << "Saddle Point:         Field value = " << g[1] << std::endl;
    for(std::size_t i=0; i<numDims; i++)
      std::cout << "Saddle Point:         Coord[" << i << "] = " << g[2+i] << std::endl;
  }

  if( outputFilename.length() > 0) {
    std::fstream out;
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << "# Final image points" << std::endl;
    for(std::size_t i=0; i<nImagePts; i++) {
      out << i << " " << imagePts[i].coords[0] << " " << imagePts[i].coords[1]
	  << " " << imagePts[i].value << std::endl;
    }    
    out.close();
  }

  if(debugFilename.length() > 0) fDebug.close();

  return;
}

void
QCAD::SaddleValueResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double current_time,
		bool sum_derivs,
		const Epetra_Vector* xdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Epetra_MultiVector* Vxdot,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* Vp,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  // Evaluate response g
  if (g != NULL) evaluateResponse(current_time, xdot, x, p, *g);

  if (gx != NULL)
    gx->PutScalar(0.0);
  
  if (gp != NULL)
    gp->PutScalar(0.0);
}

void
QCAD::SaddleValueResponseFunction::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dp)
{
  // Evaluate response g
  if (g != NULL) evaluateResponse(current_time, xdot, x, p, *g);

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->PutScalar(0.0);

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}

void 
QCAD::SaddleValueResponseFunction::
postProcessResponses(const Epetra_Comm& comm, const Teuchos::RCP<Epetra_Vector>& g)
{
}

void 
QCAD::SaddleValueResponseFunction::
postProcessResponseDerivatives(const Epetra_Comm& comm, const Teuchos::RCP<Epetra_MultiVector>& gt)
{
}

/*void QCAD::SaddleValueResponseFunction::
setupBoundary(Teuchos::ParameterList& params)
{
  //Extension to non-lateral volumes will require computing distances btwn points and segement of a plane,
  //  which isn't so much more difficult than the current point-to-line-segment distance, but isn't needed yet.
  TEUCHOS_TEST_FOR_EXCEPTION (bLateralVolumes == false, Teuchos::Exceptions::InvalidParameter, std::endl 
             << "Saddle Point for non-lateral volume is not supported yet." << std::endl); 

  std::string domain = params.get<std::string>("Domain", "box");

  if(domain == "box") {
    // Assume at least two dimensions (see constructor) and lateral volumes (above) 
    //  so create a rectangular polygon from xmin,xmax,ymin,ymax:
    double xmin, ymin, xmax, ymax;

    xmin = params.get<double>("x min");
    xmax = params.get<double>("x max");
    ymin = params.get<double>("y min");
    ymax = params.get<double>("y max");

    // move clockwise from xmin,ymin point
    nebBoundaryPiece bdPiece(numDims);

    bdPiece.p1[0] = xmin; bdPiece.p1[1] = ymin; 
    bdPiece.p2[0] = xmin; bdPiece.p2[1] = ymax; 
    boundary.push_back(bdPiece);

    bdPiece.p1[0] = xmin; bdPiece.p1[1] = ymax; 
    bdPiece.p2[0] = xmax; bdPiece.p2[1] = ymax; 
    boundary.push_back(bdPiece);

    bdPiece.p1[0] = xmax; bdPiece.p1[1] = ymax; 
    bdPiece.p2[0] = xmax; bdPiece.p2[1] = ymin; 
    boundary.push_back(bdPiece);


    bdPiece.p1[0] = xmax; bdPiece.p1[1] = ymin; 
    bdPiece.p2[0] = xmin; bdPiece.p2[1] = ymin; 
    boundary.push_back(bdPiece);

    //Add allowed z-range if in 3D (lateral volume assumed)
    if(numDims > 2) {
      zmin = params.get<double>("z min");
      zmax = params.get<double>("z max");
    }
  }
  //Element Block domain not implemented yet - will need to find bounding box for element block to use as edges
  //else if(domain == "element block") {
  //  ebName = plist->get<string>("Element Block Name");
  //}
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
             << "Error!  Invalid domain type " << domain << std::endl); 

  // Initialize boundary minima structure
  boundaryMinima.resize( boundary.size() );
  for(std::size_t i=0; i<boundaryMinima.size(); i++) 
    boundaryMinima[i].init(numDims);

  return;
}
*/


std::string QCAD::SaddleValueResponseFunction::getMode()
{
  return mode;
}


//Returns the boundary index (>=0) if point lies on 
// a boundary, otherwise -1
/*int QCAD::SaddleValueResponseFunction::
checkIfPointIsOnBoundary(const double* p)
{
  double d;
  for(std::size_t i=0; i<boundary.size(); i++) {
    d = distanceFromLineSegment(p, boundary[i].p1.data(), 
				boundary[i].p2.data(), numDims);
    if(d < imagePtSize) return i;
  }
  return -1;
  }*/

bool QCAD::SaddleValueResponseFunction::
pointIsInImagePtRegion(const double* p)
{
  //assumes at least 2 dimensions
  if(numDims > 2 && (p[2] < zmin || p[2] > zmax)) return false;
  return !(p[0] < xmin || p[1] < ymin || p[0] > xmax || p[1] > ymax);
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
    if( ptInPolygon(beginPolygon, p) ) {
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
    if( ptInPolygon(endPolygon, p) ) {
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
  double w;
  for(std::size_t i=0; i<nImagePts; i++) {
    w = pointFn( imagePts[i].coords.distanceTo(p) );
    if(w > 0) {
      imagePtWeights[i] += w;
      imagePtValues[i] += w*value;
      for(std::size_t k=0; k<numDims; k++)
	imagePtGradComps[k*nImagePts+i] += w*grad[k];
    }
  }
  return;
}

//Adds and returns the weight of a point relative to the saddle point position.
double QCAD::SaddleValueResponseFunction::
getSaddlePointWeight(const double* p)
{
  return pointFn( saddlePt.distanceTo(p) );
}

double QCAD::SaddleValueResponseFunction::
pointFn(double d) {
  const double N = 1.0;
  double val = N*exp(-d*d / (2*imagePtSize*imagePtSize));
  return (val >= 1e-2) ? val : 0.0;
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

// Returns true if point is inside polygon, false otherwise
//  Assumes 2D points (more dims ok, but uses only first two components)
//  Algorithm = ray trace along positive x-axis
bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const QCAD::mathVector& pt) 
{
  return ptInPolygon(polygon, pt.data());
}

bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const double* pt)
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


//Not used - but keep for reference
// Returns true if point is inside polygon, false otherwise
bool orig_ptInPolygon(int npol, float *xp, float *yp, float x, float y)
{
  int i, j; bool c = false;
  for (i = 0, j = npol-1; i < npol; j = i++) {
    if ((((yp[i] <= y) && (y < yp[j])) ||
	 ((yp[j] <= y) && (y < yp[i]))) &&
	(x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
      c = !c;
  }
  return c;
}







/*************************************************************/
//! mathVector class: a vector with math operations (helper class) 
/*************************************************************/

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

