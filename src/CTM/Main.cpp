#include "CTM_Solver.hpp"

#include <Albany_APFMeshStruct.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>

bool TpetraBuild = true;

int main(int argc, char** argv) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);
  auto comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  assert(argc == 2);
  auto input = argv[1];
  auto params = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::updateParametersFromXmlFile(input, params.ptr());

  {
    auto solver = rcp(new CTM::Solver(comm, params));
    solver->solve();
  }

  Albany::APFMeshStruct::finalize_libraries();
  Kokkos::finalize();
}
