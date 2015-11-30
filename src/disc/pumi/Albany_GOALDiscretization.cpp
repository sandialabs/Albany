//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_GOALDiscretization.hpp"

Albany::GOALDiscretization::GOALDiscretization(
    Teuchos::RCP<Albany::GOALMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  PUMIDiscretization(meshStruct_, commT_, rigidBodyModes_)
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

void Albany::GOALDiscretization::computeOwnedNodesAndUnknowns()
{
  apf::FieldShape* s = goalMeshStruct->getShape();
  computeOwnedNodesAndUnknownsBase(s);
}

void Albany::GOALDiscretization::computeOverlapNodesAndUnknowns()
{
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
      goalNodeSets[NS_name].push_back(gn);
    }
  }
}

void Albany::GOALDiscretization::computeSideSets()
{
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
  apf::FieldShape* s = goalMeshStruct->getShape();
  apf::FieldShape* ms = m->getShape();
  apf::changeMeshShape(m, s);
  updateMeshBase(shouldTransferIPData);
  apf::changeMeshShape(m, ms);
}
