//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_GOALDiscretization.hpp"
#include <apfShape.h>

Albany::GOALDiscretization::GOALDiscretization(
    Teuchos::RCP<Albany::GOALMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  PUMIDiscretization(meshStruct_, commT_, rigidBodyModes_),
  vtxNumbering(0)
{
  goalMeshStruct = meshStruct_;
  init();
}

Albany::GOALDiscretization::~GOALDiscretization()
{
}

int Albany::GOALDiscretization::getNumNodesPerElem(int ebi)
{
  return goalMeshStruct->getNumNodesPerElem(ebi);
}

void Albany::GOALDiscretization::setupMLCoords()
{
  return;
}

const Teuchos::ArrayRCP<double>&
Albany::GOALDiscretization::getCoordinates() const
{
  const int spdim = getNumDim();
  coordinates.resize(spdim * numOverlapVertices);
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();
  for (size_t i = 0; i < vertices.getSize(); ++i)
  {
    if (spdim == 3)
      apf::getComponents(f, vertices[i].entity, vertices[i].node,
          &coordinates[3*i]);
    else {
      double buf[3];
      apf::getComponents(f, vertices[i].entity, vertices[i].node, buf);
      double* const c = &coordinates[spdim*i];
      for (int j = 0; j < spdim; ++j) c[j] = buf[j];
    }
  }
  return coordinates;
}

void Albany::GOALDiscretization::computeOwnedNodesAndUnknowns()
{
  apf::FieldShape* s = goalMeshStruct->getShape();
  computeOwnedNodesAndUnknownsBase(s);
}

void Albany::GOALDiscretization::computeOverlapNodesAndUnknowns()
{
  apf::Mesh* m = goalMeshStruct->getMesh();
  if (vtxNumbering) apf::destroyNumbering(vtxNumbering);
  vtxNumbering = apf::numberOverlapNodes(m, "vtx", m->getShape());
  apf::getNodes(vtxNumbering, vertices);
  numOverlapVertices = vertices.getSize();
  apf::FieldShape* s = goalMeshStruct->getShape();
  computeOverlapNodesAndUnknownsBase(s);
}

void Albany::GOALDiscretization::computeGraphs()
{
  apf::FieldShape* s = goalMeshStruct->getShape();
  computeGraphsBase(s);
}

void Albany::GOALDiscretization::computeWorksetInfo()
{
  // bng: I think computeWorksetInfoBase may be
  // filling in the coords structure with garbage
  // need to look into that
  apf::FieldShape* s = goalMeshStruct->getShape();
  computeWorksetInfoBase(s);
}

void Albany::GOALDiscretization::computeNodeSets()
{
  // make sure all maps are allocated
  for (int i=0; i < meshStruct->nsNames.size(); ++i)
  {
    std::string name = meshStruct->nsNames[i];
    goalNodeSets[name].resize(0);
  }
  // grab the analsis model and mesh
  apf::StkModels& sets = meshStruct->getSets();
  apf::Mesh* m = meshStruct->getMesh();
  // loop over mesh nodes
  for (size_t i=0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    apf::MeshEntity* e = node.entity;
    if (!m->isOwned(e))
      continue;
    bool higherOrder = true;
    if (m->getType(e) == apf::Mesh::VERTEX)
      higherOrder = false;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(m, sets.invMaps[0], m->toModel(e), mset);
    if (mset.empty())
      continue;
    GO node_gid = apf::getNumber(globalNumbering, node);
    int node_lid = node_mapT->getLocalElement(node_gid);
    assert(node_lid >= 0);
    assert(node_lid < numOwnedNodes);
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* ns = *mit;
      std::string const& NS_name = ns->stkName;
      GOALNode gn;
      gn.lid = node_lid;
      gn.higherOrder = higherOrder;
      if (!higherOrder)
        m->getPoint(node.entity, node.node, gn.coord);
      goalNodeSets[NS_name].push_back(gn);
    }
  }
}

void Albany::GOALDiscretization::computeSideSets()
{
}

void Albany::GOALDiscretization::createAdjointField()
{
  int valueType;
  int neq = goalMeshStruct->neq; 
  if (neq==1)
    valueType = apf::SCALAR;
  else if (neq == 2 || neq == 3)
    valueType = apf::VECTOR;
  else {
    assert(neq == 4 || neq == 9);
    valueType = apf::MATRIX;
  }
  std::string n = std::string(APFMeshStruct::solution_name) + "_adj";
  goalMeshStruct->createNodalField(n.c_str(), valueType);
}

void Albany::GOALDiscretization::
attachSolutionToMesh(Tpetra_Vector const& x)
{
  Teuchos::ArrayRCP<const ST> data = x.get1dView();
  if (solNames.size() == 0)
    this->setField(APFMeshStruct::solution_name, &(data[0]), false);
  else
    this->setSplitFields(solNames, solIndex, &(data[0]), false);
}

void Albany::GOALDiscretization::
attachAdjointSolutionToMesh(Tpetra_Vector const& x)
{
  assert (solNames.size() == 0);
  std::string n = std::string(APFMeshStruct::solution_name) + "_adj";
  apf::Field* f = goalMeshStruct->getMesh()->findField(n.c_str());
  if (!f)
    createAdjointField();
  Teuchos::ArrayRCP<const ST> data = x.get1dView();
  this->setField(n.c_str(), &(data[0]), false);
}

void Albany::GOALDiscretization::
fillSolutionVector(Teuchos::RCP<Tpetra_Vector>& x)
{
  Teuchos::ArrayRCP<ST> data = x->get1dViewNonConst();
  if (solNames.size() == 0)
    this->getField(APFMeshStruct::solution_name, &(data[0]), true);
  else
    this->getSplitFields(solNames, solIndex, &(data[0]), true);
}

void Albany::GOALDiscretization::updateMesh(bool shouldTransferIPData)
{
  apf::Mesh* m = goalMeshStruct->getMesh();
  updateMeshBase(shouldTransferIPData);
}
