//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_Discretization.hpp"
#include "apf.h"
#include "apfMesh.h"
#include "apfShape.h"
#include "Albany_StateManager.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_AbstractPUMIDiscretization.hpp"

namespace GOAL {

using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcp_dynamic_cast;
using Albany::StateManager;
using Albany::PUMIMeshStruct;
using Albany::AbstractDiscretization;
using Albany::AbstractPUMIDiscretization;

static void getDiscretization(
    RCP<StateManager>& sm,
    RCP<AbstractPUMIDiscretization>& d,
    apf::Mesh** m)
{
  RCP<AbstractDiscretization> ad = sm->getDiscretization();
  d = rcp_dynamic_cast<AbstractPUMIDiscretization>(ad);
  RCP<PUMIMeshStruct> ms = d->getPUMIMeshStruct();
  *m = ms->getMesh();
}

static void getSolutionFields(
    RCP<AbstractPUMIDiscretization>& d,
    apf::Mesh* m,
    std::vector<std::string>& solNames,
    std::vector<int>& solIndex,
    ArrayRCP<apf::Field*>& fields)
{
  solNames = d->getSolNames();
  solIndex = d->getSolIndex();
  if (solNames.size() == 0)
    solNames.push_back(PUMIMeshStruct::solution_name);
  if (solIndex.size() == 0)
    solIndex.push_back(m->getDimension());
  fields.resize(solNames.size());
  for (int i=0; i<solNames.size(); ++i)
    fields[i] = m->findField(solNames[i].c_str());
}

static void changeMeshShape(int o, apf::Mesh* m,
    RCP<AbstractPUMIDiscretization>& d)
{
  apf::changeMeshShape(m,apf::getLagrange(o),/*project*/true);
  d->updateMesh(/*ip transfer*/false);
}

Discretization::Discretization(RCP<StateManager>& sm)
{
  getDiscretization(sm,disc,&mesh);
  getSolutionFields(disc,mesh,solNames,solIndex,solFields);
}

Discretization::~Discretization()
{
}

apf::Field* projectField(apf::Field* f)
{
  assert(apf::getShape(f) == apf::getLagrange(1));
  const char* on = apf::getName(f);
  std::string nn = std::string("e_") + std::string(on);
  apf::Field* nf = apf::createLagrangeField(
      apf::getMesh(f),nn.c_str(),apf::getValueType(f),2);
  apf::projectField(nf,f);
  return nf;
}

static void projectSolFields(
    ArrayRCP<apf::Field*> fields,
    std::vector<std::string>& solNames)
{
  for (int i=0; i < fields.size(); ++i)
  {
    fields[i] = projectField(fields[i]);
    solNames[i] = std::string(apf::getName(fields[i]));
  }
}

void Discretization::enrichDiscretization()
{
  projectSolFields(solFields,solNames);
  changeMeshShape(2,mesh,disc);
}

static void destroyEnrichedFields(
    ArrayRCP<apf::Field*> fields)
{
  for (int i=0; i < fields.size(); ++i)
    apf::destroyField(fields[i]);
}

void Discretization::decreaseDiscretization()
{
  changeMeshShape(1,mesh,disc);
  destroyEnrichedFields(solFields);
}

void Discretization::writeMesh(const char* n)
{
  apf::writeVtkFiles(n,mesh);
}

void Discretization::updateSolutionToMesh(
    const Tpetra_Vector& x)
{
  Teuchos::ArrayRCP<const ST> data = x.get1dView();
  for (int i=0; i < solFields.size(); ++i)
    disc->setSplitFields(solNames,solIndex,&(data[0]),false);
}

void Discretization::fillSolution(Teuchos::RCP<Tpetra_Vector>& x)
{
  Teuchos::ArrayRCP<ST> data = x->get1dViewNonConst();
  for (int i=0; i < solFields.size(); ++i)
    disc->getSplitFields(solNames,solIndex,&(data[0]),false);
}

}
