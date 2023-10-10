#ifndef ALBANY_OMEGAH_HPP
#define ALBANY_OMEGAH_HPP

#include "Albany_CommTypes.hpp"

#include "Omega_h_library.hpp"
#include "Teuchos_RCP.hpp"


namespace Albany {

Omega_h::Library& get_omegah_lib ();
void init_omegah_lib (int argc, char** argv,
                      const Teuchos::RCP<const Teuchos_Comm>& comm);

void finalize_omegah_lib ();

} // namespace Albany

#endif // ALBANY_OMEGAH_HPP
