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
evaluateResponses(const Epetra_Vector* xdot,
		  const Epetra_Vector& x,
		  const Teuchos::Array< Teuchos::RCP<ParamVec> >& p,
		  Epetra_Vector& g)
{

  if (!solutionLoaded) {
    EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);
    solutionLoaded = true;
  }

  Epetra_Vector diff(x.Map());
  double norm2;
  diff.Update(1.0,x,-1.0,*RefSoln,0.0); 
  diff.Norm2(&norm2);
  g[0]=norm2*norm2;
}

void
Albany::SolutionFileL2ResponseFunction::
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
  TEST_FOR_EXCEPTION(true, std::logic_error,
     "Albany::SolutionFileL2ResponseFunction:: evaluateTangents not implemented\n");
}

void
Albany::SolutionFileL2ResponseFunction::
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
  if (!solutionLoaded) {
    EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);
    solutionLoaded = true;
  }

  Epetra_Vector diff(x.Map());
  double norm2;

  // Evaluate response g
  if (g != NULL) {
    diff.Update(1.0,x,-1.0,*RefSoln,0.0);
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
  for (Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >::size_type j=0; j<dg_dp.size(); j++)
    if (dg_dp[j] != Teuchos::null)
      dg_dp[j]->PutScalar(0.0);
}

void
Albany::SolutionFileL2ResponseFunction::
evaluateSGResponses(const Stokhos::VectorOrthogPoly<Epetra_Vector>* sg_xdot,
		    const Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_x,
		    const ParamVec* p,
		    const ParamVec* sg_p,
		    const Teuchos::Array<SGType>* sg_p_vals,
		    Stokhos::VectorOrthogPoly<Epetra_Vector>& sg_g)
{
  TEST_FOR_EXCEPTION(true, std::logic_error,
     "Albany::SolutionFileL2ResponseFunction:: evaluateSGResponse  not implemented\n");
}
