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


#include <Epetra_LocalMap.h>
#include "QCAD_SaddleValueResponseFunction.hpp"


//! Helper function prototypes
void gatherVector(std::vector<double>& v, std::vector<double>& gv,
		  const Epetra_Comm& comm);
void getOrdering(const std::vector<double>& v, std::vector<int>& ordering);
bool lessOp(std::pair<std::size_t, double> const& a,
	    std::pair<std::size_t, double> const& b);
double averageOfVector(const std::vector<double>& v);
double distance(const std::vector<double>* vCoords,
		int ind1, int ind2, std::size_t nDims);
double distance(const std::vector<double>* vCoords,
		int ind1, double* pt2, std::size_t nDims);



QCAD::SaddleValueResponseFunction::
SaddleValueResponseFunction(const int numDim_, Teuchos::ParameterList& params)
  : numDims(numDim_)
{
  fieldCutoffFctr = params.get<double>("Field Cutoff Factor", 1.0);
  minPoolDepthFctr = params.get<double>("Minimum Pool Depth Factor", 0.1);
  distanceCutoffFctr = params.get<double>("Distance Cutoff Factor", 0.2);

  bRetPosOnFailGiven = (params.isParameter("Fallback X") || 
			params.isParameter("Fallback Y") ||
			params.isParameter("Fallback Z"));
  retPosOnFail[0] = params.get<double>("Fallback X", 0.0);
  retPosOnFail[1] = params.get<double>("Fallback Y", 0.0);
  retPosOnFail[2] = params.get<double>("Fallback Z", 0.0);
}

QCAD::SaddleValueResponseFunction::
~SaddleValueResponseFunction()
{
}

unsigned int
QCAD::SaddleValueResponseFunction::
numResponses() const 
{
  return 5;  // returnFieldValue, fieldValue, saddleX, saddleY, saddleZ
}

void
QCAD::SaddleValueResponseFunction::
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
  vFieldValues.clear();
  vRetFieldValues.clear();
  vCellVolumes.clear();
  
  for(std::size_t k = 0; k < numDims; k++)
    vCoords[k].clear();
}

void
QCAD::SaddleValueResponseFunction::
evaluateTangents(
	   const Epetra_Vector* xdot,
	   const Epetra_Vector& x,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	   const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dxdot_dp,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dx_dp,
	   Epetra_Vector* g,
	   const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& gt)
{
  // Evaluate response g
  if (g != NULL) evaluateResponses(xdot, x, p, *g);

  // Evaluate tangent of g = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<gt.size(); j++)
    if (gt[j] != Teuchos::null)
      for (int i=0; i<dx_dp[i]->NumVectors(); i++)
	(*gt[j])[i][0] = 0.0;
}

void
QCAD::SaddleValueResponseFunction::
evaluateGradients(
	  const Epetra_Vector* xdot,
	  const Epetra_Vector& x,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
	  const Teuchos::Array< Teuchos::RCP<ParamVec> >& deriv_p,
	  Epetra_Vector* g,
	  Epetra_MultiVector* dg_dx,
	  Epetra_MultiVector* dg_dxdot,
	  const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dg_dp)
{
  // Evaluate response g
  if (g != NULL) evaluateResponses(xdot, x, p, *g);

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->PutScalar(0.0);

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);

  // Evaluate dg/dp
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
    if (dg_dp[j] != Teuchos::null)
      dg_dp[j]->PutScalar(0.0);
}

void
QCAD::SaddleValueResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  unsigned int sz = sg_x.size();
  for (unsigned int i=0; i<sz; i++)
    sg_g[i][0] = 0.0;
}



void 
QCAD::SaddleValueResponseFunction::
postProcessResponses(const Epetra_Comm& comm, Teuchos::RCP<Epetra_Vector>& g)
{
  bool bShowInfo = (comm.MyPID() == 0);

  //! Gather data from different processors
  std::vector<double> allFieldVals;
  std::vector<double> allRetFieldVals;
  std::vector<double> allCellVols;
  std::vector<double> allCoords[MAX_DIMENSION];

  gatherVector(vFieldValues, allFieldVals, comm);  
  gatherVector(vRetFieldValues, allRetFieldVals, comm);
  gatherVector(vCellVolumes, allCellVols, comm);

  for(std::size_t k=0; k<numDims; k++)
    gatherVector(vCoords[k], allCoords[k], comm);

  //! Exit early if there are no field values in the specified region
  if( allFieldVals.size()  == 0 ) {
    for(std::size_t k=0; k<5; k++) (*g)[k] = 0;
    return;
  }

  //! Print gathered size on proc 0
  if(bShowInfo) {
    std::cout << std::endl << "--- Begin Saddle Point Response Function ---" << std::endl;
    std::cout << "--- Saddle: local size (this proc) = " << vFieldValues.size()
	      << ", gathered size (all procs) = " << allFieldVals.size() << std::endl;
  }

  //! Sort data by field value
  std::vector<int> ordering;
  getOrdering(allFieldVals, ordering);


  //! Compute max/min for distance and field value
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
  
  //double maxDistanceDelta = 0.0;
  //for(std::size_t k=0; k<numDims && k < 3; k++) {
  //  if( fabs(maxCoords[k] - minCoords[k]) > maxDistanceDelta )
  //    maxDistanceDelta = fabs(maxCoords[k] - minCoords[k]);
  //}
  double avgCellLength = pow(averageOfVector(allCellVols), 1.0/numDims);
  double maxFieldDifference = fabs(maxFieldVal - minFieldVal);

  if(bShowInfo) {
    std::cout << "--- Saddle: max field difference = " << maxFieldDifference
	      << ", avg cell length = " << avgCellLength << std::endl;
  }

  //! Set cutoffs
  double cutoffDistance, cutoffFieldVal, minDepth;
  cutoffDistance = avgCellLength * distanceCutoffFctr;
  cutoffFieldVal = maxFieldDifference * fieldCutoffFctr;
  minDepth = maxFieldDifference * minPoolDepthFctr;

  int result, nRestarts=0; 
  do {
    result = FindSaddlePoint(allFieldVals, allRetFieldVals, allCoords, ordering,
			     cutoffDistance, cutoffFieldVal, minDepth, bShowInfo, g);
    if(result == 1) { //failed b/c not enough deep pools
      if(minDepth > 0) {
	minDepth /= 2;
	if(bShowInfo) std::cout << "--- Saddle: RESTARTING with min depth = "
				<< minDepth << std::endl;
      }
      else break;
    }
    else if(result == 2)  //failed because not enough pools
      cutoffDistance *= 2;

    nRestarts++;
  } while(result != 0 && nRestarts <= 10);  //i.e while FindSaddlePoint failed

  if(result != 0) {
    if(bRetPosOnFailGiven) {
      double d, minDist=1e80;
      int minDistIndex = 0;
      for(std::size_t i=0; i < N; i++) {
	d = distance(allCoords, i, retPosOnFail, numDims);
	if( d < minDist ) { minDist = d; minDistIndex = i; }
      }
      
      // Return values at cell closest to retPosOnFail,
      //  even though a saddle point has not been found
      (*g)[0] = allRetFieldVals[minDistIndex];
      (*g)[1] = allFieldVals[minDistIndex];
      for(std::size_t k=0; k<numDims && k < 3; k++)
	(*g)[2+k] = allCoords[k][minDistIndex];

      if(bShowInfo) std::cout << "--- Saddle not found: "
			      << "returning user point values.";
    }
    else {
      for(std::size_t k=0; k<5; k++) (*g)[k] = 0; //output all zeros
      if(bShowInfo) std::cout << "--- Saddle not found.";
    }
  }

  return;
}


//! Level-set Algorithm for finding saddle point
int QCAD::SaddleValueResponseFunction::
FindSaddlePoint(std::vector<double>& allFieldVals, std::vector<double>& allRetFieldVals,
		std::vector<double>* allCoords, std::vector<int>& ordering,
		double cutoffDistance, double cutoffFieldVal, double minDepth, bool bShortInfo,
		Teuchos::RCP<Epetra_Vector>& g)
{
  bool bDebug = false;

  if(bShortInfo) {
    std::cout << "--- Saddle: distance cutoff = " << cutoffDistance
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

    if(bDebug || bShortInfo) {
      nDeepTrees = 0;
      for(std::size_t t=0; t < treeSizes.size(); t++) {
	if(treeSizes[t] > 0 && (allFieldVals[I]-minFieldVals[t]) > minDepth) nDeepTrees++;
      }
    }

    if(bDebug) std::cout << "DEBUG: i=" << i << "( I = " << I << "), val="
			 << allFieldVals[I] << ", loc=(" << allCoords[0][I] 
			 << "," << allCoords[1][I] << ")" << " nD=" << nDeepTrees;

    if(bShortInfo && lastDeepTrees != nDeepTrees) {
      std::cout << "--- Saddle: i=" << i << " new deep pool: nPools=" << nTrees 
		<< " nDeep=" << nDeepTrees << std::endl;
      lastDeepTrees = nDeepTrees;
    }

    for(int j=i-1; fabs(allFieldVals[I] - allFieldVals[ordering[j]]) < cutoffFieldVal && j >= 0; j--) {
      J = ordering[j];

      if( distance(allCoords, I, J, numDims) < cutoffDistance ) {
	if(treeIDs[I] == -1) {
	  treeIDs[I] = treeIDs[J];
	  treeSizes[treeIDs[I]]++;

	  if(bDebug) std::cout << " --> tree " << treeIDs[J] 
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

	  if(bDebug) std::cout << "DEBUG:   also --> " << treeIDs[J] 
			       << " [merged] size=" << treeSizes[treeIDs[J]]
			       << " (treecount after merge = " << nTrees << ")" << std::endl;

	  if(bShortInfo) std::cout << "--- Saddle: i=" << i << "merge: nPools=" << nTrees 
				   << " nDeep=" << nDeepTrees << std::endl;


	  if(mergingTwoDeepTrees && nDeepTrees == 1) {
	    if(bDebug) std::cout << "DEBUG: FOUND SADDLE! exiting." << std::endl;
	    if(bShortInfo) std::cout << "--- Saddle: i=" << i << " Found saddle." << std::endl;

	    //Found saddle at I
	    (*g)[0] = allRetFieldVals[I];
	    (*g)[1] = allFieldVals[I];
	    for(std::size_t k=0; k<numDims && k < 3; k++)
	      (*g)[2+k] = allCoords[k][I];

	    return 0; //success
	  }

	}
      }

    } //end j loop
    
    if(treeIDs[I] == -1) {
      if(bDebug) std::cout << " --> new tree with ID " << nextAvailableTreeID
			   << " (treecount after new = " << (nTrees+1) << ")" << std::endl;
      if(bShortInfo) std::cout << "--- Saddle: i=" << i << " new pool: nPools=" << (nTrees+1) 
			       << " nDeep=" << nDeepTrees << std::endl;

      treeIDs[I] = nextAvailableTreeID++;
      minFieldVals.push_back(allFieldVals[I]);
      treeSizes.push_back(1);

      nTrees += 1;
      if(nTrees > nMaxTrees) nMaxTrees = nTrees;
    }

  } // end i loop

  // if no saddle found, return all zeros
  if(bDebug) std::cout << "DEBUG: NO SADDLE. exiting." << std::endl;
  for(std::size_t k=0; k<5; k++) (*g)[k] = 0;

  // if two or more trees where found, then reason for failure is that not
  //  enough deep pools were found - so could try to reduce minDepth and re-run.
  if(nMaxTrees >= 2) return 1;

  // nMaxTrees < 2 - so we need more trees.  Could try to increase cutoffDistance and/or cutoffFieldVal.
  return 2;
}

void 
QCAD::SaddleValueResponseFunction::
postProcessResponseDerivatives(const Epetra_Comm& comm, Teuchos::RCP<Epetra_MultiVector>& gt)
{
}

void 
QCAD::SaddleValueResponseFunction::
addFieldData(double fieldValue, double retFieldValue, double* coords, double cellVolume)
{ 
  //std::cout << "DEBUG: Adding field data: " << fieldValue << ", " << retFieldValue 
  //	    << ", pt = ( " << coords[0] << ", " << coords[1] << " )" << std::endl;

  vFieldValues.push_back(fieldValue);
  vRetFieldValues.push_back(retFieldValue);
  vCellVolumes.push_back(cellVolume);

  for(std::size_t i=0; i < numDims; ++i)
    vCoords[i].push_back(coords[i]);
}




/*************************************************************/
//! Helper functions
/*************************************************************/

void gatherVector(std::vector<double>& v, std::vector<double>& gv, const Epetra_Comm& comm)
{
  double *pvec, zeroSizeDummy = 0;
  pvec = (v.size() > 0) ? &v[0] : &zeroSizeDummy;

  Epetra_Map map(-1, v.size(), 0, comm);
  Epetra_Vector ev(View, map, pvec);
  int  N = map.NumGlobalElements();
  Epetra_LocalMap lomap(N,0,comm);

  gv.resize(N);
  pvec = (gv.size() > 0) ? &gv[0] : &zeroSizeDummy;
  Epetra_Vector egv(View, lomap, pvec);
  Epetra_Import import(lomap,map);
  egv.Import(ev, import, Insert);
}

bool lessOp(std::pair<std::size_t, double> const& a,
	    std::pair<std::size_t, double> const& b) {
  return a.second < b.second;
}

void getOrdering(const std::vector<double>& v, std::vector<int>& ordering)
{
  typedef std::vector<double>::const_iterator dbl_iter;
  typedef std::vector<std::pair<std::size_t, double> >::const_iterator pair_iter;
  std::vector<std::pair<std::size_t, double> > vPairs(v.size());

  size_t n = 0;
  for (dbl_iter it = v.begin(); it != v.end(); ++it, ++n)
    vPairs[n] = std::make_pair(n, *it);


  std::sort(vPairs.begin(), vPairs.end(), lessOp);

  ordering.resize(v.size()); n = 0;
  for (pair_iter it = vPairs.begin(); it != vPairs.end(); ++it, ++n)
    ordering[n] = it->first;
}


double averageOfVector(const std::vector<double>& v)
{
  double avg = 0.0;
  for(std::size_t i=0; i < v.size(); i++) {
    avg += v[i];
  }
  avg /= v.size();
  return avg;
}

double distance(const std::vector<double>* vCoords, int ind1, int ind2, std::size_t nDims)
{
  double d2 = 0;
  for(std::size_t k=0; k<nDims; k++)
    d2 += pow( vCoords[k][ind1] - vCoords[k][ind2], 2 );
  return sqrt(d2);
}

double distance(const std::vector<double>* vCoords, int ind1, double* pt2, std::size_t nDims)
{
  double d2 = 0;
  for(std::size_t k=0; k<nDims; k++)
    d2 += pow( vCoords[k][ind1] - pt2[k], 2 );
  return sqrt(d2);
}

