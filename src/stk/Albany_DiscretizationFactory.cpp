//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#ifdef ALBANY_SEACAS
#include "Albany_IossSTKMeshStruct.hpp"
#endif
#ifdef ALBANY_CUTR
#include "Albany_FromCubitSTKMeshStruct.hpp"
#endif


Albany::DiscretizationFactory::DiscretizationFactory(
	    const Teuchos::RCP<Teuchos::ParameterList>& discParams_, bool adaptive,
               const Teuchos::RCP<const Epetra_Comm>& epetra_comm_) :
  discParams(discParams_), adaptiveMesh(adaptive), epetra_comm(epetra_comm_)
{
}

#ifdef ALBANY_CUTR
void
Albany::DiscretizationFactory::setMeshMover(const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover_)
{
  meshMover = meshMover_;
}
#endif

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
Albany::DiscretizationFactory::createMeshSpecs()
{
  std::string& method = discParams->get("Method", "STK1D");
  if (method == "STK1D") {
    stkMeshStruct = Teuchos::rcp(new Albany::TmplSTKMeshStruct<1>(discParams, adaptiveMesh, epetra_comm));
    //stkMeshStruct = Teuchos::rcp(new Albany::Line1DSTKMeshStruct(discParams, epetra_comm));
  }
  else if (method == "STK0D") {
    stkMeshStruct = Teuchos::rcp(new Albany::TmplSTKMeshStruct<0>(discParams, adaptiveMesh, epetra_comm));
  }
  else if (method == "STK2D") {
    stkMeshStruct = Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(discParams, adaptiveMesh, epetra_comm));
  }
  else if (method == "STK3D") {
    stkMeshStruct = Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(discParams, adaptiveMesh, epetra_comm));
  }
  else if (method == "Ioss" || method == "Exodus" ||  method == "Pamgen") {
#ifdef ALBANY_SEACAS
    stkMeshStruct = Teuchos::rcp(new Albany::IossSTKMeshStruct(discParams, adaptiveMesh, epetra_comm));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" ||  method == "Pamgen",
          Teuchos::Exceptions::InvalidParameter,
         "Error: Discretization method " << method 
          << " requested, but not compiled in" << std::endl);
#endif
  }
  else if (method == "Cubit") {
#ifdef ALBANY_CUTR
    AGS"need to inherit from Generic"
    stkMeshStruct = Teuchos::rcp(new Albany::FromCubitSTKMeshStruct(meshMover, discParams, neq));
#else 
    TEUCHOS_TEST_FOR_EXCEPTION(method == "Cubit", 
          Teuchos::Exceptions::InvalidParameter,
         "Error: Discretization method " << method 
          << " requested, but not compiled in" << std::endl);
#endif
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl << 
       "Error!  Unknown discretization method in DiscretizationFactory: " << method << 
       "!" << std::endl << "Supplied parameter list is " << std::endl << *discParams 
       << "\nValid Methods are: STK1D, STK2D, STK3D, Ioss, Exodus, Cubit" << std::endl);
  }

  return stkMeshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(unsigned int neq,
                           const Teuchos::RCP<Albany::StateInfoStruct>& sis)
{
  TEUCHOS_TEST_FOR_EXCEPTION(stkMeshStruct==Teuchos::null,
       std::logic_error,
       "stkMeshStruct accessed, but it has not been constructed" << std::endl);

  stkMeshStruct->setFieldAndBulkData(epetra_comm, discParams, neq,
                                     sis, stkMeshStruct->getMeshSpecs()[0]->worksetSize);

  Teuchos::RCP<Albany::AbstractDiscretization> strategy
    = Teuchos::rcp(new Albany::STKDiscretization(stkMeshStruct, epetra_comm));

  return strategy;
}
