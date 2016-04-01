#ifndef ATO_TYPES_DEF
#define ATO_TYPES_DEF

#include "ATO_TopoTools.hpp"
#include <unordered_map>

namespace ATO {

typedef struct TopologyStruct {
  Teuchos::RCP<ATO::Topology> topology;
  Teuchos::RCP<Epetra_Vector> dataVector;
} TopologyStruct;

}

#endif

