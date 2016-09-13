#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>
#include <PCU.h>

#include <Albany_DataTypes.hpp>
#include <Albany_DiscretizationFactory.hpp>
#include <Albany_APFMeshStruct.hpp>

static void validate_parameters(
    Teuchos::RCP<const Teuchos::ParameterList> p) {
  assert(p->isSublist("Discretization"));
}

int main(int argc, char** argv) {

  // initialize the parallel services
  Teuchos::GlobalMPISession mpiSession(&argc,&argv,NULL);
  Kokkos::initialize(argc, argv);
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);

  // get the default communicator object
  Teuchos::RCP<const Teuchos_Comm> comm =
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  // get the input file to run this problem
  assert(argc == 2);
  auto input = argv[1];
  auto params = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::updateParametersFromXmlFile(input, params.ptr());
  validate_parameters(params);

  // create a discretization factory
  bool explicit_time_method = false;
  auto disc_factory = Teuchos::rcp(new Albany::DiscretizationFactory(
        params, comm, explicit_time_method));

  // finalize the parallel services.
  Albany::APFMeshStruct::finalize_libraries();
  Kokkos::finalize();
}
