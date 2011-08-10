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


QCAD::SaddleValueResponseFunction::
SaddleValueResponseFunction(const int numDim_)
  : numDims(numDim_)
{
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

//helper function
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

void
QCAD::SaddleValueResponseFunction::
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{
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
  //Gather data from different processors
  std::vector<double> allFieldVals;
  std::vector<double> allRetFieldVals;
  std::vector<double> allCellVols;
  std::vector<double> allCoords[MAX_DIMENSION];

  gatherVector(vFieldValues, allFieldVals, comm);  
  gatherVector(vRetFieldValues, allRetFieldVals, comm);
  gatherVector(vCellVolumes, allCellVols, comm);

  for(std::size_t k=0; k<numDims; k++)
    gatherVector(vCoords[k], allCoords[k], comm);


  // check for case that there are no field values in the specified region
  std::size_t N = allFieldVals.size();
  if( N  == 0 ) {
    for(std::size_t k=0; k<5; k++) (*g)[k] = 0;
    return;
  }

  //std::cout << "DEBUG: Response Function -- mySize = " << vFieldValues.size()
  //	    << ", gatheredSize = " << allFieldVals->MyLength() << std::endl;

  //DEBUG TEST - set responses to average quantities
  /*(*g)[0] = averageOfVector(allRetFieldVals);
  (*g)[1] = averageOfVector(allFieldVals);
  for(std::size_t k=0; k<numDims && k < 3; k++) // 3 hardcoded b/c g has 5 elements (see above)
    (*g)[2+k] = averageOfVector(allCoords[k]);
  */

  // Level-set Algorithm for finding saddle point

  // Sort data by field value
  std::vector<int> ordering;
  getOrdering(allFieldVals, ordering);

  // Compute thresholds for distance and field value
  double maxFieldVal = allFieldVals[0], minFieldVal = allFieldVals[0];
  double maxCoords[3], minCoords[3];

  for(std::size_t k=0; k<numDims && k < 3; k++)
    maxCoords[k] = minCoords[k] = allCoords[k][0];

  for(std::size_t i=0; i<N; i++) {
    for(std::size_t k=0; k<numDims && k < 3; k++) {
      if(allCoords[k][i] > maxCoords[k]) maxCoords[k] = allCoords[k][i];
      if(allCoords[k][i] < minCoords[k]) minCoords[k] = allCoords[k][i];
    }
    if(allFieldVals[i] > maxFieldVal) maxFieldVal = allFieldVals[i];
    if(allFieldVals[i] < minFieldVal) minFieldVal = allFieldVals[i];
  }

  
  double maxDistanceDelta = 0.0;
  for(std::size_t k=0; k<numDims && k < 3; k++) {
    if( fabs(maxCoords[k] - minCoords[k]) > maxDistanceDelta )
      maxDistanceDelta = fabs(maxCoords[k] - minCoords[k]);
  }

  double cutoffDistance, cutoffFieldVal;
  cutoffDistance = maxDistanceDelta / 5; //hardcoded - variable later
  cutoffFieldVal = fabs(maxFieldVal - minFieldVal) / 50; //hardcoded - variable later
  std::cout << "DEBUG: distance cutoff = " << cutoffDistance
	    << ", field cutoff = " << cutoffFieldVal << std::endl;

  // Walk through sorted data.  At current point, walk backward in list 
  //  until either 1) a "close" point is found, as given by tolerance -> join to tree
  //            or 2) the change in field value exceeds some maximium -> new tree

  std::cout << "DEBUG: begin algorithm" << std::endl;
  std::vector<int> treeIDs(N, -1);
  int nTrees = 0, nextAvailableTreeID = 1, treeIDtoReplace;
  int I, J, K;
  for(std::size_t i=0; i < N; i++) {
    I = ordering[i];
    std::cout << "DEBUG: i=" << i << "( I = " << I << "), val="
	      << allFieldVals[I] << ", loc=(" << allCoords[0][I] 
	      << "," << allCoords[1][I] << ")" << std::endl;

    for(int j=i-1; fabs(allFieldVals[I] - allFieldVals[ordering[j]]) < cutoffFieldVal && j >= 0; j--) {
      J = ordering[j];
      std::cout << "DEBUG:   j=" << j << "( J = " << J << "), val="
	      << allFieldVals[J] << ", loc=(" << allCoords[0][J] 
	      << "," << allCoords[1][J] << ")" << std::endl;


      if( distance(allCoords, I, J, numDims) < cutoffDistance ) {

	std::cout << "DEBUG:   > j=" << j << " close to i=" << i 
		  << " : treeIDs = " << treeIDs[J] << "," << treeIDs[I] << std::endl;

	if(treeIDs[I] == -1) {
	  treeIDs[I] = treeIDs[J];
	}
	else if(treeIDs[I] != treeIDs[J]) {
	  std::cout << "DEBUG:   > merging trees " << treeIDs[I] << " --> " << treeIDs[J]
		    << " (treecount after merge = " << (nTrees-1) << ")" << std::endl;

	  treeIDtoReplace = treeIDs[I];
	  for(int k=i; k >=0; k--) {
	    K = ordering[k];
	    if(treeIDs[K] == treeIDtoReplace)
	      treeIDs[K] = treeIDs[J];
	  }
	  nTrees -= 1;


	  if(nTrees == 1) {
	    std::cout << "DEBUG: FOUND SADDLE! exiting." << std::endl;

	    //Found saddle at I
	    (*g)[0] = allRetFieldVals[I];
	    (*g)[1] = allFieldVals[I];
	    for(std::size_t k=0; k<numDims && k < 3; k++)
	      (*g)[2+k] = allCoords[k][I];

	    return;
	  }

	}
      }

    } //end j loop
    
    if(treeIDs[I] == -1) {
      std::cout << "DEBUG: creating new tree with ID " << nextAvailableTreeID
		<< " (treecount after new = " << (nTrees+1) << ")" << std::endl;
      treeIDs[I] = nextAvailableTreeID++;
      nTrees += 1;
    }

  } // end i loop

  // if no saddle found, return all zeros
  std::cout << "DEBUG: NO SADDLE. exiting." << std::endl;
  for(std::size_t k=0; k<5; k++) (*g)[k] = 0;
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
