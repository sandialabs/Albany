#include <Omega_h_mesh.hpp>

namespace OmegahGhost {
  Omega_h::LO getNumOwnedElms(Omega_h::Mesh& mesh);
  Omega_h::HostRead<Omega_h::GO> getOwnedElementGids(Omega_h::Mesh& mesh, int dim);
  Omega_h::Read<Omega_h::I8> getEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
  Omega_h::LO getNumEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
}
