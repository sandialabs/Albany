#include <Omega_h_mesh.hpp>

namespace OmegahGhost {
  Omega_h::LO getNumOwnedElms(const Omega_h::Mesh& cmesh);
  Omega_h::HostRead<Omega_h::GO> getOwnedEntityGids(const Omega_h::Mesh& cmesh, int dim);
  Omega_h::Read<Omega_h::I8> getEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
  Omega_h::HostRead<Omega_h::GO> getEntGidsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
  Omega_h::Reals getVtxCoordsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh);
  Omega_h::LO getNumEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
  Omega_h::LOs getDownAdjacentEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim);
}
