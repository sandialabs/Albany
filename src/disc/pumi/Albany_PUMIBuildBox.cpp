//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIMeshStruct.hpp"
#include <apfBox.h>
#include <sstream>

namespace Albany {

static void buildSets(apf::BoxBuilder& bb, apf::StkModels& sets)
{
  int nsides = bb.dim * 2;
  /* arrange these however you expect
     the NodeSets and SideSets to be ordered */
  apf::Indices const faceTable[6] = {
    apf::Indices(0,1,1),
    apf::Indices(2,1,1),
    apf::Indices(1,0,1),
    apf::Indices(1,2,1),
    apf::Indices(1,1,0),
    apf::Indices(1,1,2),
  };
  int dims[2] = {0, bb.dim - 1};
  char const* names[2] = {"NodeSet", "SideSet"};
  for (int i = 0; i < 2; ++i) {
    sets.models[dims[i]].resize(nsides);
    for (int j = 0; j < nsides; ++j) {
      apf::Indices mi = faceTable[j];
      for (int k = bb.dim; k < 3; ++k)
        mi[k] = 0;
      int mj = bb.mgrid.in(mi);
      apf::StkModel* set = new apf::StkModel();
      int mdim = bb.modelTable[mj].dim;
      int mtag = bb.modelTable[mj].tag;
      apf::ModelEntity* me = bb.m->findModelEntity(mdim, mtag);
      set->ents.push_back(me);
      std::stringstream ss;
      ss << names[i] << j;
      set->stkName = ss.str();
      sets.models[dims[i]][j] = set;
    }
  }
  sets.models[bb.dim].push_back(new apf::StkModel());
  sets.models[bb.dim][0]->ents.push_back(bb.m->findModelEntity(bb.dim, 0));
  sets.models[bb.dim][0]->stkName = "Block0";
}

void PUMIMeshStruct::buildBoxMesh(
    int nex, int ney, int nez,
    double wx, double wy, double wz, bool is)
{
  apf::BoxBuilder bb(nex, ney, nez, wx, wy, wz, is);
  mesh = bb.m;
  buildSets(bb, sets);
}

}
