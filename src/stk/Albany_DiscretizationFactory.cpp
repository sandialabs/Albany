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


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Point0DSTKMeshStruct.hpp"
#include "Albany_Line1DSTKMeshStruct.hpp"
#include "Albany_Rect2DSTKMeshStruct.hpp"
#include "Albany_Cube3DSTKMeshStruct.hpp"
#ifdef ALBANY_IOSS
#include "Albany_IossSTKMeshStruct.hpp"
#endif
#ifdef ALBANY_CUTR
#include "Albany_FromCubitSTKMeshStruct.hpp"
#endif


Albany::DiscretizationFactory::DiscretizationFactory(
	    const Teuchos::RCP<Teuchos::ParameterList>& discParams_) :
  discParams(discParams_)
{
}

#ifdef ALBANY_CUTR
void
Albany::DiscretizationFactory::setMeshMover(const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover_)
{
  meshMover = meshMover_;
}
#endif

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::create(
		  unsigned int neq,
		  unsigned int nstates,
                  const Teuchos::RCP<const Epetra_Comm>& epetra_comm)
{
  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct;

  std::string& method = discParams->get("Method", "STK1D");
  if (method == "STK1D") {
    
    stkMeshStruct = Teuchos::rcp(new Albany::Line1DSTKMeshStruct(epetra_comm, discParams, neq, nstates));

  }
  else if (method == "STK0D") {
    
    stkMeshStruct = Teuchos::rcp(new Albany::Point0DSTKMeshStruct(epetra_comm, discParams, neq, nstates));

  }
  else if (method == "STK2D") {
    
    stkMeshStruct = Teuchos::rcp(new Albany::Rect2DSTKMeshStruct(epetra_comm, discParams, neq, nstates));

  }
  else if (method == "STK3D") {
    
    stkMeshStruct = Teuchos::rcp(new Albany::Cube3DSTKMeshStruct(epetra_comm, discParams, neq, nstates));

  }
  else if (method == "Ioss" || method == "Exodus" ||  method == "Pamgen") {
#ifdef ALBANY_IOSS
    
    stkMeshStruct = Teuchos::rcp(new Albany::IossSTKMeshStruct(epetra_comm, discParams, neq, nstates));

#else
    TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" ||  method == "Pamgen",
          Teuchos::Exceptions::InvalidParameter,
         "Error: Discretization method " << method 
          << " requested, but not compiled in" << std::endl);
#endif
  }
  else if (method == "Cubit") {
#ifdef ALBANY_CUTR
    
    stkMeshStruct = Teuchos::rcp(new Albany::FromCubitSTKMeshStruct(meshMover, discParams, neq, nstates));
#else 
    TEST_FOR_EXCEPTION(method == "Cubit", 
          Teuchos::Exceptions::InvalidParameter,
         "Error: Discretization method " << method 
          << " requested, but not compiled in" << std::endl);
#endif
  }
  else {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl << 
       "Error!  Unknown discretization method in DiscretizationFactory: " << method << 
       "!" << std::endl << "Supplied parameter list is " << std::endl << *discParams 
       << "\nValid Methods are: STK1D, STK2D, STK3D, Ioss, Exodus, Cubit" << std::endl);
  }

  Teuchos::RCP<Albany::AbstractDiscretization> strategy
    = Teuchos::rcp(new Albany::STKDiscretization(stkMeshStruct, epetra_comm));

  return strategy;
}
