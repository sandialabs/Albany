#ifndef ATO_TYPES_HPP
#define ATO_TYPES_HPP

#include <ATO_TopoTools.hpp>

#include "Albany_config.h"
#include "Albany_ThyraTypes.hpp"

#include <mpi.h>

namespace ATO {

struct TopologyStruct {
  Teuchos::RCP<Topology>      topology;
  Teuchos::RCP<Thyra_Vector>  dataVector;
};

using ATO_GO = GO;

struct GlobalPoint {

  // LB: I initially did 'GlobalPoint() = default;', but that
  //     does not 0-initialize the coordinates, which 'can'
  //     cause problems when checking if two points are the same
  GlobalPoint() : gid(-1) {
    coords[0] = coords[1] = coords[2] = 0.0;
  }

  ATO_GO  gid;
  double  coords[3];
};

inline bool operator< (GlobalPoint const & a, GlobalPoint const & b) {
  return a.gid < b.gid;
}

inline MPI_Datatype get_MPI_GlobalPoint_type () {
  MPI_Datatype MPI_GlobalPoint;

  auto GO_MPI_INT_TYPE = sizeof(ATO_GO)==sizeof(std::int32_t) ? MPI_INT32_T : MPI_INT64_T;

  // this should go somewhere else.  for now ...
  GlobalPoint gp;
  int blockcounts[2] = {1,3};
  MPI_Datatype oldtypes[2] = {GO_MPI_INT_TYPE, MPI_DOUBLE};
  MPI_Aint offsets[3] = {(MPI_Aint)&(gp.gid)    - (MPI_Aint)&gp, 
                         (MPI_Aint)&(gp.coords) - (MPI_Aint)&gp};
  MPI_Type_create_struct(2,blockcounts,offsets,oldtypes,&MPI_GlobalPoint);
  MPI_Type_commit(&MPI_GlobalPoint);

  return MPI_GlobalPoint;
}

} // namespace ATO

#endif // ATO_TYPES_HPP
