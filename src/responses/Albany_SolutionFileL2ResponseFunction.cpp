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


#include "Albany_SolutionFileL2ResponseFunction.hpp"
#include "EpetraExt_VectorIn.h"

#include "Epetra_Map.h"  
#include "EpetraExt_BlockMapIn.h"  
#include "Epetra_SerialComm.h"  ///HAQ

Albany::SolutionFileL2ResponseFunction::
SolutionFileL2ResponseFunction()
  : RefSoln(NULL), solutionLoaded(false)
{
}

Albany::SolutionFileL2ResponseFunction::
~SolutionFileL2ResponseFunction()
{
  if (solutionLoaded) delete RefSoln;
}

unsigned int
Albany::SolutionFileL2ResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionFileL2ResponseFunction::
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{

  int MMFileStatus = 0;

  // Read the reference solution for comparison from "reference_solution.dat"

  // Note that this is of MatrixMarket array real general format

  if (!solutionLoaded) {
    MMFileStatus = EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus, std::runtime_error,
      std::endl << "EpetraExt::MatrixMarketFileToVector, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }


  // Build a vector to hold the difference between the actual and reference solutions
  Epetra_Vector diff(x.Map());

  double norm2;

  // The diff vector equals 1.0 * soln + -1.0 * reference
  diff.Update(1.0,x,-1.0,*RefSoln,0.0); 

  // Print vector for debugging
  // diff.Print(std::cout);

  // Get the 2 norm
  diff.Norm2(&norm2);

  g[0]=norm2*norm2;
}

void
Albany::SolutionFileL2ResponseFunction::
evaluateTangent(
	   const double alpha, 
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
  Teuchos::RCP<Epetra_MultiVector> dgdx;
  if (gx != NULL && Vx != NULL)
    dgdx = Teuchos::rcp(new Epetra_MultiVector(x.Map(), 1));
  else
    dgdx = Teuchos::rcp(gx,false);
  evaluateGradient(current_time, xdot, x, p, deriv_p, g, dgdx.get(), NULL, gp);
  if (gx != NULL && Vx != NULL)
    gx->Multiply('T', 'N', alpha, *dgdx, *Vx, 0.0);
}

void
Albany::SolutionFileL2ResponseFunction::
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
  int MMFileStatus = 0;

  if (!solutionLoaded) {
    MMFileStatus = EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus, std::runtime_error,
      std::endl << "EpetraExt::MatrixMarketFileToVector, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }


  // Build a vector to hold the difference between the actual and reference solutions
  Epetra_Vector diff(x.Map());

  double norm2;

  // Evaluate response g
  if (g != NULL) {

  // The diff vector equals 1.0 * soln + -1.0 * reference

    diff.Update(1.0,x,-1.0,*RefSoln,0.0);

    // Print vector for debugging
    // diff.Print(std::cout);

    diff.Norm2(&norm2);
    (*g)[0]=norm2*norm2;
  }

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->Update(2.0,x,-2.0,*RefSoln,0.0);

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
