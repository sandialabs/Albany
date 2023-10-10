#include "Albany_Omegah.hpp"
#include "Albany_CommUtils.hpp"

namespace Albany {

std::shared_ptr<Omega_h::Library>&
get_omegah_lib_impl ()
{
  static std::shared_ptr<Omega_h::Library> lib;
  return lib;
}


Omega_h::Library&
get_omegah_lib ()
{
  auto lib = get_omegah_lib_impl();
  TEUCHOS_TEST_FOR_EXCEPTION (lib==nullptr, std::logic_error,
      "Error! Omega_h lib was not initialized.\n");
  return *lib;
}

void init_omegah_lib (int argc, char** argv,
                      const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  auto& lib = get_omegah_lib_impl();
  TEUCHOS_TEST_FOR_EXCEPTION (lib!=nullptr, std::logic_error,
      "Error! You are supposed to initialize Omega_h library only once!\n");

  auto mpi_comm = getMpiCommFromTeuchosComm(comm);
  lib = std::make_shared<Omega_h::Library>(&argc,&argv,mpi_comm);
}

void finalize_omegah_lib ()
{
  auto& lib = get_omegah_lib_impl();

  // Should we error out if lib==nullptr already? I don't see a downside
  // to allow multiple calls to finalize, so I'm not going to error out here.
  lib = nullptr;
}

} // namespace Albany
