#include <Albany_APFMeshStruct.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>
#include <PCU.h>

#include "CTM_Solver.hpp"

// sad global variable that must be declared :(
bool TpetraBuild = true;

int main(int argc, char** argv) {

    // initialize the parallel services
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
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

    // build the solver and solve the problem
    // This bracket are needed for now just to avoid a segmentation  fault
    // when the program exit. I want all the RCP pointers created in solver
    // get out of scope before we call APFMeshStruct::finalize_libraries
    {
        auto solver = rcp(new CTM::Solver(comm, params));
        solver->solve();
    }
    // finalize the parallel services.
    Albany::APFMeshStruct::finalize_libraries();
    Kokkos::finalize();
}
