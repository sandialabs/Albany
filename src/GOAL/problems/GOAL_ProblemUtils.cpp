//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_ProblemUtils.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_ParameterList.hpp"

namespace GOAL {

void enrichMeshSpecs(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > ms)
{
  int physSets = ms.size();
  for (int ps=0; ps < physSets; ++ps)
  {
    const char* name = ms[ps]->ctd.name;
    assert(strcmp(name, "Tetrahedron_4") == 0);
    const CellTopologyData* ctd =
      shards::getCellTopologyData<shards::Tetrahedron<10> >();
    ms[ps]->ctd = *ctd;
  }
}

void decreaseMeshSpecs(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > ms)
{
  int physSets = ms.size();
  for (int ps=0; ps < physSets; ++ps)
  {
    const char* name = ms[ps]->ctd.name;
    assert(strcmp(name, "Tetrahedron_10") == 0);
    const CellTopologyData* ctd =
      shards::getCellTopologyData<shards::Tetrahedron<4> >();
    ms[ps]->ctd = *ctd;
  }
}

Teuchos::RCP<Teuchos::ParameterList> getAdjointDirichletBCs(
    Teuchos::RCP<Teuchos::ParameterList>& bcs)
{
  Teuchos::RCP<Teuchos::ParameterList> p = rcp(new Teuchos::ParameterList);
  return Teuchos::null;
}

}
