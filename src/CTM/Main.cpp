#include "CTM_Solver.hpp"

#include <Albany_APFMeshStruct.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>
#include <Albany_Utils.hpp>

int main(int argc, char** argv) {

  static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);
  auto comm = Tpetra::getDefaultComm();

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
