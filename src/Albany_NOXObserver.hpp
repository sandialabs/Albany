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


#ifndef ALBANY_NOXOBSERVER
#define ALBANY_NOXOBSERVER


#include "Albany_VTK.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "NOX_Epetra_Observer.H"

class Albany_NOXObserver : public NOX::Epetra::Observer
{
public:
   Albany_NOXObserver (
         const Teuchos::RCP<Albany_VTK> vtk_,
         const Teuchos::RCP<Albany::AbstractDiscretization> &disc_);

   ~Albany_NOXObserver ()
   { };

  void observeSolution(
    const Epetra_Vector& solution);

private:

   Teuchos::RCP<Albany::AbstractDiscretization> disc;
   Teuchos::RCP<Albany_VTK> vtk;

};

#endif //ALBANY_NOXOBSERVER
