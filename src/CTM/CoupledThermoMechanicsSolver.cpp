#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>
#include <PCU.h>

int main(int argc, char** argv) {
  Teuchos::GlobalMPISession mpiSession(&argc,&argv,NULL);
  Kokkos::initialize(argc, argv);
  PCU_Comm_Init();
  PCU_Comm_Free();
  Kokkos::finalize();
}
