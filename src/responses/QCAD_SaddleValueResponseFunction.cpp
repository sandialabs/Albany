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
#include "QCAD_SaddleValueResponseFunction.hpp"


//! Helper function prototypes
double distanceFromLineSegment(const double* p, const double* p1, const double* p2, int dims);


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
  imagePtSize   = params.get<double>("Image Point Size", 0.01);
  nImagePts     = params.get<int>("Number of Image Points", 10);
  maxIterations = params.get<int>("Maximum Iterations", 100);
  timeStep      = params.get<double>("Time Step", 1.0);
  baseSpringConstant   = params.get<double>("Base Spring Constant", 1.0);
  convergenceThreshold = params.get<double>("Convergence Threshold", 1e-3);

  /*Teuchos::Array<double> beginPt;
  beginPt = params.get<Teuchos::Array<double> >("Begin Point");
  for(int i=0; i<beginPt.size(); i++) {
    std::cout << "BEGIN POINT: " << beginPt[i] << std::endl;
    }*/
  //beginRegionType = params.get<std::string>("Begin Region Type");
  //endRegionType   = params.get<std::string>("End Region Type");
  //if(beginRegionType == "Point")

    //else
    //beginEB = params.get<std::string>("Begin Element Block");

      

  debugMode = params.get<int>("Debug Mode",0);
  bPositiveOnly = params.get<bool>("Positive Return Only",false);
  bLateralVolumes = true; //TODO - add as a param

  imagePts.resize(nImagePts);

  setupBoundary(params);

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
  bool bClimbing = true;

  // Find saddle point in stages:

  // 1) Determine initial and final points which lie on the edge
  //      of the bounding polygon (now just a 2D polygon since lateral volume required)
  
  if(dbMode > 1) std::cout << "Saddle Point:  Beginning minimization over boundary" << std::endl;
  mode = "Minima on boundary";
  Albany::FieldManagerScalarResponseFunction::evaluateResponse(
	current_time, xdot, x, p, g);
  if(dbMode > 2) std::cout << "Saddle Point:   -- done evaluation" << std::endl;

    //MPI: get global min for each boundary piece
  double* localMinima = new double[boundaryMinima.size()];
  double* globalMinima = new double[boundaryMinima.size()];
  for(std::size_t i=0; i<boundaryMinima.size(); i++)
    localMinima[i] = boundaryMinima[i].value;
  comm.MinAll( localMinima, globalMinima, boundaryMinima.size() );

  for(std::size_t i=0; i<boundaryMinima.size(); i++) {
    int procToBcast;
    if( fabs(boundaryMinima[i].value - globalMinima[i]) < 1e-8 ) 
      procToBcast = comm.MyPID();
    else procToBcast = -1;

    int winner;
    comm.MaxAll( &procToBcast, &winner, 1 );
    comm.Broadcast( boundaryMinima[i].coords.data(), numDims, winner); //broadcast winner's min position to others
    boundaryMinima[i].value = globalMinima[i];                         //no need to broadcast min value
  }

  if(dbMode > 2) std::cout << "Saddle Point:   -- done MPI communication" << std::endl;

  delete [] localMinima; localMinima = NULL;
  delete [] globalMinima; globalMinima = NULL;


    // Choose initial point as minimum of minimums along boundary pieces,
    //  then choose final point as next lowest minimum from among the 
    //  rest of the boundary pieces *not adjacent* to the piece with the global min.
  int imin = 0;
  for(std::size_t i=1; i<boundaryMinima.size(); i++) {
    if( boundaryMinima[i].value < boundaryMinima[imin].value ) imin = (int)i;
  }

  int imin2 = (imin == 0) ? 2 : 0;
  for(int i=0; i <= imin-2; i++) {
    if( boundaryMinima[i].value < boundaryMinima[imin2].value ) imin2 = i;
  }
  for(std::size_t i=imin+2; i < boundaryMinima.size(); i++) {
    if( boundaryMinima[i].value < boundaryMinima[imin2].value ) imin2 = i;
  }

  mathVector& initialPt = boundaryMinima[imin].coords;
  mathVector& finalPt = boundaryMinima[imin2].coords;
  mathVector pt(numDims);

    // Initialize Image Points:  interpolate between initial and final points to get all the image points
  TEUCHOS_TEST_FOR_EXCEPTION (nImagePts < 2, Teuchos::Exceptions::InvalidParameter, std::endl 
	      << "Saddle Point needs more than 2 image pts (" << nImagePts << " given)" << std::endl); 

  imagePts[0].init(initialPt);
  imagePts[nImagePts-1].init(finalPt);

  for(std::size_t i=1; i<nImagePts-1; i++) {
    double s = i * 1.0/(nImagePts-1);   // nIntervals = nImagePts-1
    imagePts[i].init(initialPt + (finalPt - initialPt) * s);
  }
    // Note: all procs compute the same image points - no need for MPI communication here
      
  if(dbMode > 1) {
    for(std::size_t i=0; i<nImagePts; i++)
      std::cout << "Saddle Point:   -- imagePt[" << i << "] = " << imagePts[i].coords << std::endl;
  }

  //  2) Perform Nudged Elastic Band Algorithm to find saddle point.
  //      Iterate over field manager fills of each image point's value and gradient
  //       then update image point positions user Verlet algorithm
  
  mode = "Collect image point data";
  std::size_t nIters;
  mathVector tangent(numDims), springForce(numDims), dNext(numDims), dPrev(numDims);
  double mag, dp, maxUpdate, dValuePrev, dValueNext;
  std::vector<mathVector> force(nImagePts);
  std::vector<double> springConstants(nImagePts-1, baseSpringConstant);
  for(std::size_t i=0; i<nImagePts; i++) force[i].resize(numDims);

  double dt = timeStep;
  double dt2 = dt*dt;
  nIters = 0;

  
  //Allocate storage for aggrecated image point data (needed for MPI)
  double*  imagePtValues  = new double [nImagePts];
  double*  imagePtWeights = new double [nImagePts];
  double*  imagePtGrads   = new double [nImagePts*numDims];
  double*  globalPtValues   = new double [nImagePts];
  double*  globalPtWeights  = new double [nImagePts];
  double*  globalPtGrads    = new double [nImagePts*numDims];
  double*  lastImagePtValues  = new double [nImagePts];

  while( ++nIters <= maxIterations) {

    if(dbMode > 1) std::cout << "Saddle Point:  NEB Algorithm iteration " << nIters << " -----------------------" << std::endl;

    //Reset value, weight, and gradient of image points as these are accumulated by evaluator fill
    for(std::size_t i=0; i<nImagePts; i++) {
      lastImagePtValues[i] = imagePts[i].value;
      imagePts[i].value = imagePts[i].weight = 0.0;
      imagePts[i].grad.fill(0.0);
    }

    Albany::FieldManagerScalarResponseFunction::evaluateResponse(
				    current_time, xdot, x, p, g);

    //MPI -- sum weights, value, and gradient for each image pt
    for(std::size_t i=0; i<nImagePts; i++) {
      imagePtValues[i]  = imagePts[i].value;
      imagePtWeights[i] = imagePts[i].weight;
      for(std::size_t k=0; k<numDims; k++) imagePtGrads[k*nImagePts+i] = imagePts[i].grad[k];
    }
    comm.SumAll( imagePtValues,  globalPtValues,    nImagePts );
    comm.SumAll( imagePtWeights, globalPtWeights, nImagePts );
    comm.SumAll( imagePtGrads,   globalPtGrads,   nImagePts*numDims );

    // Put summed data back std::size_to imagePts
    for(std::size_t i=0; i<nImagePts; i++) {
      imagePts[i].value = globalPtValues[i];
      imagePts[i].weight = globalPtWeights[i];
      for(std::size_t k=0; k<numDims; k++) imagePts[i].grad[k] = globalPtGrads[k*nImagePts+i];
    }

    // Normalize value and gradient from different cell contributions
    for(std::size_t i=1; i<nImagePts-1; i++) {
      if( imagePts[i].weight > 1e-6 ) {
	imagePts[i].value /= imagePts[i].weight;
	imagePts[i].grad  /= imagePts[i].weight;
      }
      else { //assume point has drifted off region: reset value and set gradient to zero
	imagePts[i].value = lastImagePtValues[i];
	imagePts[i].grad.fill(0.0);
      }
    }

    // Find highest point if using the climbing NEB technique
    int iHighestPt = 0; //effectively a null since loops below begin at 1
    if(bClimbing && nIters > 20) {
      for(std::size_t i=2, iHightestPt=1; i<nImagePts-1; i++) {
	if(imagePts[i].value > imagePts[iHighestPt].value) iHightestPt = i;
      }
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
	

      if((int)i == iHighestPt) {
	// Special case for highest point in climbing-NEB: force has full -Grad(V) but with parallel 
	//    component reversed and no force from springs
	dp = imagePts[i].grad.dot( tangent );
	force[i] = (imagePts[i].grad * -1.0 + (tangent*dp) * 2); // force += -Grad(V) + 2*Grad(V)_parallel

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
	force[i] -= (imagePts[i].grad - tangent * dp); // force += -Grad(V)_perp

	if(dbMode > 2) {
	  std::cout << "Saddle Point:  --   tangent = " << tangent << std::endl;
	  std::cout << "Saddle Point:  --   grad along tangent = " << dp << std::endl;
	  std::cout << "Saddle Point:  --   grad force = " << force[i] << std::endl;
	}

	// Get spring force projected parallel to the tangent and add to the force
	dPrev = imagePts[i-1].coords - imagePts[i].coords;
	dNext = imagePts[i+1].coords - imagePts[i].coords;
	springForce = tangent * ((dNext.norm() * springConstants[i]) - (dPrev.norm() * springConstants[i-1]));
	force[i] += springForce;  // force += springForce_parallel

	if(dbMode > 2) {
	  std::cout << "Saddle Point:  --   spring force = " << springForce << std::endl;
	  std::cout << "Saddle Point:  --   total force = " << force[i] << std::endl;
	}
      }

      if(dbMode > 2) std::cout << "Saddle Point:  --   Force on pt[" << i << "] = " << imagePts[i].value 
				  << " : " << imagePts[i].coords << " is " << force[i] << std::endl;
    }

    maxUpdate=0;
    
    //Update points using precomputed forces
    for(std::size_t i=1; i<nImagePts-1; i++) {

      //Update coordinates using Verlet integration
      mathVector dCoords = imagePts[i].velocity * dt + force[i] * dt2;
      imagePts[i].coords += dCoords;

      //Update velocity similarly, but reset to zero if current velocity 
      // is directed opposite to force (reduces overshoot)
      dp = imagePts[i].velocity.dot(force[i]);
      if(dp >= 0) imagePts[i].velocity += force[i] * dt;
      else imagePts[i].velocity.fill(0.0);  

      mag = dCoords.norm();
      if(mag > maxUpdate) maxUpdate = mag;
    }

    if(dbMode > 1) std::cout << "Saddle Point:  >> Max Update magnitude " << maxUpdate 
			    << " < " << convergenceThreshold << std::endl;

    //Check convergence
    if(maxUpdate < convergenceThreshold) break;    
  }

  //deallocate storage used for MPI communication
  delete [] imagePtValues;  
  delete [] imagePtWeights; 
  delete [] imagePtGrads;   
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

void QCAD::SaddleValueResponseFunction::
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


std::string QCAD::SaddleValueResponseFunction::getMode()
{
  return mode;
}


//Returns the boundary index (>=0) if point lies on 
// a boundary, otherwise -1
int QCAD::SaddleValueResponseFunction::
checkIfPointIsOnBoundary(const double* p)
{
  double d;
  for(std::size_t i=0; i<boundary.size(); i++) {
    d = distanceFromLineSegment(p, boundary[i].p1.data(), 
				boundary[i].p2.data(), numDims);
    if(d < imagePtSize) return i;
  }
  return -1;
}

void QCAD::SaddleValueResponseFunction::
addBoundaryData(const double* p, double value)
{
  int iBd = checkIfPointIsOnBoundary(p);
  if( iBd >= 0 ) { 
    if( value < boundaryMinima[iBd].value ) {
      boundaryMinima[iBd].value = value;
      boundaryMinima[iBd].coords.fill(p);
      if(debugMode) std::cout << "DEBUG: found point on boundary edge " << iBd << std::endl;
    }
  }
  return;
}


bool QCAD::SaddleValueResponseFunction::
checkIfPointIsWithinBoundary(const double* p)
{
  //Note: assumes lateral volume with at least 2 dimensions
  bool c = false;
  double x = p[0], y = p[1];

  // check z-coordinate if its present
  if(numDims > 2) {  
    if(p[2] > zmax || p[2] < zmin) return false;
  }

  // check that x,y is within 2D boundary
  for(std::size_t i=0; i<boundary.size(); i++) {
    if ((((boundary[i].p1[1] <= y) && (y < boundary[i].p2[1])) ||
	 ((boundary[i].p2[1] <= y) && (y < boundary[i].p1[1]))) &&
	(x < (boundary[i].p2[0] - boundary[i].p1[0]) * (y - boundary[i].p1[1]) /
	 (boundary[i].p2[1] - boundary[i].p1[1]) + boundary[i].p1[0]))
      c = !c;
  }
  return c;
}


void QCAD::SaddleValueResponseFunction::
addImagePointData(const double* p, double value, double* grad)
{
  double d, w, N=1.0;
  for(std::size_t i=0; i<nImagePts; i++) {
    d = imagePts[i].coords.distanceTo(p);
    w = N*exp(-d*d / (2*imagePtSize*imagePtSize));
    if(w > 1e-2) {
      imagePts[i].weight += w;
      imagePts[i].value += w*value;
      for(std::size_t k=0; k<numDims; k++)
	imagePts[i].grad[k] += w*grad[k];
    }
  }
  return;
}

//Adds and returns the weight of a point relative to the saddle point position.
double QCAD::SaddleValueResponseFunction::
getSaddlePointWeight(const double* p)
{
  double w, d = saddlePt.distanceTo(p), N=1.0;
  w = N*exp(-d*d / (2*imagePtSize*imagePtSize));
  if(w > 1e-2) return w;
  else return 0.0;
}


/*************************************************************/
//! Helper functions
/*************************************************************/

double distanceFromLineSegment(const double* p, const double* p1, const double* p2, int dims)
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
}

//Not used - but keep for reference
// Returns true if point is inside polygon, false otherwise
bool ptInPolygon(int npol, float *xp, float *yp, float x, float y)
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

void 
QCAD::mathVector::normalize() 
{
  (*this) /= norm(); 
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

std::ostream& QCAD::operator<<(std::ostream& os, const QCAD::nebPt& np)
{
  os << std::endl;
  os << "coords = " << np.coords << std::endl;
  os << "veloc  = " << np.velocity << std::endl;
  os << "grad   = " << np.grad << std::endl;
  os << "value  = " << np.value << std::endl;
  os << "weight = " << np.weight << std::endl;
  return os;
}

