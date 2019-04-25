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

  GlobalPoint() = default;

  ATO_GO  gid;
  double  coords[3];
};

inline bool operator< (GlobalPoint const & a, GlobalPoint const & b) {
  return a.gid < b.gid;
}

inline MPI_Datatype get_MPI_GlobalPoint_type () {
  MPI_Datatype MPI_GlobalPoint;

  // this should go somewhere else.  for now ...
  GlobalPoint gp;
  int blockcounts[2] = {1,3};
  MPI_Datatype oldtypes[2] = {MPI_INT, MPI_DOUBLE};
  MPI_Aint offsets[3] = {(MPI_Aint)&(gp.gid)    - (MPI_Aint)&gp, 
                         (MPI_Aint)&(gp.coords) - (MPI_Aint)&gp};
  MPI_Type_create_struct(2,blockcounts,offsets,oldtypes,&MPI_GlobalPoint);
  MPI_Type_commit(&MPI_GlobalPoint);

  return MPI_GlobalPoint;
}

} // namespace ATO

#endif // ATO_TYPES_HPP
