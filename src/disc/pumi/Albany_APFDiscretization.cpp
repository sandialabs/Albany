//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_APFDiscretization.hpp"

#include <limits>
#if defined(ALBANY_EPETRA)
#include "Epetra_Export.h"
#endif

#include "Albany_Utils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#ifdef ALBANY_EPETRA
#include "Petra_Converters.hpp"
#endif
#include "Albany_PUMIOutput.hpp"
#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <PHAL_Dimension.hpp>

#include <apfMesh.h>
#include <apfShape.h>
#include <PCU.h>

#if defined(ALBANY_EPETRA)
// Some integer-type converter helpers for Epetra_Map so that we can compile
// the Epetra_Map file regardless of the value of ALBANY_64BIT_INT.
namespace {
typedef int EpetraInt;
#ifdef ALBANY_64BIT_INT
Teuchos::RCP< Teuchos::Array<int> >
convert (const Teuchos::Array<GO>& indicesAV) {
  Teuchos::RCP< Teuchos::Array<int> > ind = Teuchos::rcp(
    new Teuchos::Array<int>(indicesAV.size()));
  for (std::size_t i = 0; i < indicesAV.size(); ++i)
    (*ind)[i] = Teuchos::as<int>(indicesAV[i]);
  return ind;
};
#else // not ALBANY_64BIT_INT
Teuchos::RCP< Teuchos::Array<GO> >
convert (Teuchos::Array<GO>& indicesAV) {
  return Teuchos::rcp(&indicesAV, false);
}
#endif // not ALBANY_64BIT_INT
} // namespace
#endif // ALBANY_EPETRA

Albany::APFDiscretization::APFDiscretization(Teuchos::RCP<Albany::APFMeshStruct> meshStruct_,
            const Teuchos::RCP<const Teuchos_Comm>& commT_,
            const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  commT(commT_),
  rigidBodyModes(rigidBodyModes_),
  neq(meshStruct_->neq),
  meshStruct(meshStruct_),
  interleavedOrdering(meshStruct_->interleavedOrdering),
  outputInterval(0),
  continuationStep(0)
{
}

Albany::APFDiscretization::~APFDiscretization()
{
  delete meshOutput;
  apf::destroyGlobalNumbering(globalNumbering);
  apf::destroyGlobalNumbering(elementNumbering);
}

void Albany::APFDiscretization::init()
{
  meshOutput = PUMIOutput::create(meshStruct, commT);
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(commT);
#endif
  globalNumbering = 0;
  elementNumbering = 0;

  // Initialize the mesh and all data structures
  bool shouldTransferIPData = false;
  this->updateMesh(shouldTransferIPData);

// layout[num deriv vectors][DOF_component]
  Teuchos::Array<Teuchos::Array<std::string> > layout = meshStruct->solVectorLayout;
  int number_of_solution_vecs = layout.size();
  solLayout.resize(number_of_solution_vecs);


  for (std::size_t i=0; i < layout[0].size(); i+=2) {

    std::string res_name = layout[0][i];
    res_name.append("Res");

    resNames.push_back(res_name);

  }

  for (int j=0; j < number_of_solution_vecs; j++) {
    int total_ndofs = 0;
    for (std::size_t i = 0; i < layout[j].size(); i += 2) {
      solLayout.getDerivNames(j).push_back(layout[j][i]);
      int ndofs = 0;
      if (layout[j][i + 1] == "S") {
        ndofs = 1;
      } else if (layout[j][i + 1] == "V") {
        ndofs = getNumDim();
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "Layout '" << layout[j][i+1] << "' is not supported.");
      }
      solLayout.getDerivSizes(j).push_back(ndofs);
      total_ndofs += ndofs;
    }
    if (layout[0].size()) {
      TEUCHOS_TEST_FOR_EXCEPTION(total_ndofs != neq, std::logic_error,
          "Layout size " << total_ndofs <<
          " does not match number of equations " << neq << '\n');
    }
  }

  // zero the residual field for Rhythmos
  if (resNames.size())
    for (size_t i = 0; i < resNames.size(); ++i)
      apf::zeroField(meshStruct->getMesh()->findField(resNames[i].c_str()));
  else
    apf::zeroField(
      meshStruct->getMesh()->findField(APFMeshStruct::residual_name));

  // set all of the restart fields here
  if (meshStruct->hasRestartSolution)
    setRestartData();
}

Teuchos::RCP<const Tpetra_Map>
Albany::APFDiscretization::getMapT() const
{
  return mapT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::APFDiscretization::getOverlapMapT() const
{
  return overlap_mapT;
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::APFDiscretization::getOverlapNodeMap() const
{
  return Petra::TpetraMap_To_EpetraMap(overlap_node_mapT, comm);
}
#endif

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::APFDiscretization::getJacobianGraphT() const
{
  return graphT;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::APFDiscretization::getOverlapJacobianGraphT() const
{
  return overlap_graphT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::APFDiscretization::getNodeMapT() const
{
  return node_mapT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::APFDiscretization::getOverlapNodeMapT() const
{
  return overlap_node_mapT;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
Albany::APFDiscretization::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
Albany::APFDiscretization::getWsElNodeID() const
{
  return wsElNodeID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
Albany::APFDiscretization::getCoords() const
{
  return coords;
}

void
Albany::APFDiscretization::printCoords() const
{
  int mesh_dim = meshStruct->getMesh()->getDimension();

  std::cout << "Processor " << PCU_Comm_Self() << " has " << coords.size()
      << " worksets." << std::endl;

  for (int ws=0; ws<coords.size(); ws++)  //workset
    for (int e=0; e<coords[ws].size(); e++) //cell
      for (int j=0; j<coords[ws][e].size(); j++) //node
        for (int d=0; d<mesh_dim; d++) //dim
          std::cout << "Coord for workset: " << ws << " element: " << e
              << " node: " << j << " DOF: " << d << " is: " <<
              coords[ws][e][j][d] << std::endl;
}

const Teuchos::ArrayRCP<double>&
Albany::APFDiscretization::getCoordinates() const
{
  const int spdim = getNumDim();
  coordinates.resize(spdim * numOverlapNodes);
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    if (spdim == 3)
      apf::getComponents(f, nodes[i].entity, nodes[i].node, &coordinates[3*i]);
    else {
      double buf[3];
      apf::getComponents(f, nodes[i].entity, nodes[i].node, buf);
      double* const c = &coordinates[spdim*i];
      for (int j = 0; j < spdim; ++j) c[j] = buf[j];
    }
  }
  return coordinates;
}

void
Albany::APFDiscretization::setCoordinates(
    const Teuchos::ArrayRCP<const double>& c)
{
  const int spdim = getNumDim();
  double buf[3] = {0};
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    if (spdim == 3)
      apf::setComponents(f, nodes[i].entity, nodes[i].node, &c[spdim*i]);
    else {
      const double* const cp = &c[spdim*i];
      for (int j = 0; j < spdim; ++j) buf[j] = cp[j];
      apf::setComponents(f, nodes[i].entity, nodes[i].node, buf);
    }
  }
}

void Albany::APFDiscretization::
setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& ircm)
{ rcm = ircm; }

const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
Albany::APFDiscretization::getSphereVolume() const
{
  return sphereVolume;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type&
Albany::APFDiscretization::getLatticeOrientation() const
{
  return latticeOrientation;
}

double mean (const double* x, const int n,
             const Teuchos::RCP<const Tpetra_Map>& map) {
  Teuchos::ArrayView<const double> xav = Teuchos::arrayView(x, n);
  Tpetra_Vector xv(map, xav);
  return xv.meanValue();
}

/* DAI: this function also has to change for high-order fields */
void Albany::APFDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) return;
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed()) return;

  // get mesh dimension and part handle
  const int mesh_dim = getNumDim();
  coordMV = Teuchos::rcp(
      new Tpetra_MultiVector(node_mapT, mesh_dim, false));

  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();

  for (std::size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    if ( ! m->isOwned(node.entity)) continue; // Skip nodes that are not local

    const GO node_gid = apf::getNumber(globalNumbering, node);
    const LO node_lid = node_mapT->getLocalElement(node_gid);
    double lcoords[3];
    apf::getComponents(f, nodes[i].entity, nodes[i].node, lcoords);
    for (std::size_t j = 0; j < mesh_dim; ++j)
      coordMV->replaceLocalValue(node_lid, j, lcoords[j]);
  }

  if (meshStruct->useNullspaceTranslationOnly)
    rigidBodyModes->setCoordinates(coordMV);
  else
    rigidBodyModes->setCoordinatesAndNullspace(coordMV, mapT);
}

const Albany::WorksetArray<std::string>::type&
Albany::APFDiscretization::getWsEBNames() const
{
  return wsEBNames;
}

const Albany::WorksetArray<int>::type&
Albany::APFDiscretization::getWsPhysIndex() const
{
  return wsPhysIndex;
}

inline int albanyCountComponents (const int problem_dim, const int pumi_value_type) {
  switch (pumi_value_type) {
  case apf::SCALAR: return 1;
  case apf::VECTOR: return problem_dim;
  case apf::MATRIX: return problem_dim * problem_dim;
  default: assert(0); return -1;
  }
}

void Albany::APFDiscretization::setField(
  const char* name, const ST* data, bool overlapped, int offset, bool neq_sized)
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* f = m->findField(name);

  const int problem_dim = meshStruct->problemDim;
  double data_buf[9] = {0};
  const int pumi_value_type = apf::getValueType(f);
  const int albany_nc = albanyCountComponents(problem_dim, pumi_value_type);
  const int total_comps = (neq_sized ? neq : albany_nc);

  // the simple front-packing of components below would not
  // be sufficient to deal with incoming 2x2 tensors, so assert
  // that we are passing data straight through if dealing with a tensor
  if (pumi_value_type == apf::MATRIX) assert(albany_nc == 9);

  for (size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    GO node_gid = apf::getNumber(globalNumbering, node);
    int node_lid;
    if (overlapped) {
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    } else {
      if ( ! m->isOwned(node.entity)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    const int first_dof = getDOF(node_lid, offset, total_comps);

    const double* datap = data + first_dof;
    for (int j = 0; j < albany_nc; ++j) {
      data_buf[j] = datap[j];
    }

    apf::setComponents(f, node.entity, node.node, data_buf);
  }

  if ( ! overlapped)
    apf::synchronize(f);
}

void Albany::APFDiscretization::setSplitFields(
  const Teuchos::Array<std::string>& names, const Teuchos::Array<int>& sizes,
  const ST* data, bool overlapped)
{
  const int spdim = getNumDim();
  apf::Mesh* m = meshStruct->getMesh();
  int offset = 0;
  for (std::size_t i=0; i < names.size(); ++i) {
    this->setField(names[i].c_str(), data, overlapped, offset);
    offset += sizes[i];
  }
}

void Albany::APFDiscretization::getField(
  const char* name, ST* data, bool overlapped, int offset, bool neq_sized) const
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* f = m->findField(name);
  const int problem_dim = meshStruct->problemDim;
  const int pumi_value_type = apf::getValueType(f);
  const int albany_nc = albanyCountComponents(problem_dim, pumi_value_type);
  assert(albany_nc <= 3);
  const int total_comps = (neq_sized ? neq : albany_nc);
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    GO node_gid = apf::getNumber(globalNumbering,node);
    int node_lid;
    if (overlapped) {
      node_lid = overlap_node_mapT->getLocalElement(node_gid);
    } else {
      if ( ! m->isOwned(node.entity)) continue;
      node_lid = node_mapT->getLocalElement(node_gid);
    }
    const int first_dof = getDOF(node_lid, offset, total_comps);
    double buf[3];
    apf::getComponents(f, node.entity, node.node, buf);
    for (int j = 0; j < albany_nc; ++j) data[first_dof + j] = buf[j];
  }
}

void Albany::APFDiscretization::getSplitFields(
  const Teuchos::Array<std::string>& names, const Teuchos::Array<int>& sizes, ST* data,
  bool overlapped) const
{
  const int spdim = getNumDim();
  apf::Mesh* m = meshStruct->getMesh();
  int offset = 0;
  for (std::size_t i=0; i < names.size(); ++i) {
    this->getField(names[i].c_str(), data, overlapped, offset);
    offset += sizes[i];
  }
}

void Albany::APFDiscretization::reNameExodusOutput(
    const std::string& str)
{
  if (meshOutput)
    meshOutput->setFileName(str);
}

void Albany::APFDiscretization::writeSolutionT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToMeshDatabase(&(data[0]), 0, overlapped);
  writeAnySolutionToFile(time_value);
}

void Albany::APFDiscretization::writeSolutionMV(
  const Tpetra_MultiVector& solnT, const double time_value, const bool overlapped)
{

  for(int i = 0; i <= meshStruct->num_time_deriv; i++){
    Teuchos::RCP<const Tpetra_Vector> colT = solnT.getVector(i);
    Teuchos::ArrayRCP<const ST> data = colT->get1dView();
    writeAnySolutionToMeshDatabase(&(data[0]), i, overlapped);
  }

  writeAnySolutionToFile(time_value);

}

void Albany::APFDiscretization::writeSolutionToMeshDatabaseT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToMeshDatabase(&(data[0]), 0, overlapped);
}

void Albany::APFDiscretization::writeSolutionMVToMeshDatabase(
  const Tpetra_MultiVector& solnT, const double time_value, const bool overlapped)
{
  for(int i = 0; i <= meshStruct->num_time_deriv; i++){
    Teuchos::RCP<const Tpetra_Vector> colT = solnT.getVector(i);
    Teuchos::ArrayRCP<const ST> data = colT->get1dView();
    writeAnySolutionToMeshDatabase(&(data[0]), i, overlapped);
  }
}

void Albany::APFDiscretization::writeSolutionToFileT(
  const Tpetra_Vector& solnT, const double time_value, const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = solnT.get1dView();
  writeAnySolutionToFile(time_value);
}

void Albany::APFDiscretization::writeSolutionMVToFile(
  const Tpetra_MultiVector& solnT, const double time_value, const bool overlapped)
{
  for(int i = 0; i <= meshStruct->num_time_deriv; i++){

    Teuchos::RCP<const Tpetra_Vector> colT = solnT.getVector(i);
    Teuchos::ArrayRCP<const ST> data = colT->get1dView();
    writeAnySolutionToFile(time_value);
  }
}

#if defined(ALBANY_EPETRA)
void Albany::APFDiscretization::writeSolution(const Epetra_Vector& soln, const double time_value,
      const bool overlapped)
{
#if 1
  Teuchos::RCP<const Tpetra_Vector> solnT =
     Petra::EpetraVector_To_TpetraVectorConst(soln, commT);
  writeSolutionT(*solnT, time_value, overlapped);
#else
  writeAnySolutionToMeshDatabase(&(soln[0]), 0, overlapped);
  writeAnySolutionToFile(time_value);
#endif
}
#endif

static void saveOldTemperature(Teuchos::RCP<Albany::APFMeshStruct> meshStruct)
{
  if (!meshStruct->useTemperatureHack)
    return;
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* t = m->findField("temp");
  if (!t)
    t = m->findField(Albany::APFMeshStruct::solution_name[0]);
  assert(t);
  apf::Field* told = m->findField("temp_old");
  if (!told)
    told = meshStruct->createNodalField("temp_old", apf::SCALAR);
  assert(told);
  std::cout << "copying nodal " << apf::getName(t)
    << " to nodal " << apf::getName(told) << '\n';
  apf::copyData(told, t);
}

void Albany::APFDiscretization::writeAnySolutionToMeshDatabase(
      const ST* soln, const int index, const bool overlapped)
{
  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Transfer to APF Mesh");
  // time deriv vector (solution, solution_dot, or solution_dotdot)
  if (solLayout.getDerivNames(index).size() == 0) {
    this->setField(APFMeshStruct::solution_name[index], soln, overlapped);
  } else {
    this->setSplitFields(solLayout.getDerivNames(index),
                         solLayout.getDerivSizes(index),
                         soln, overlapped);
  }
  meshStruct->solutionInitialized = true;
  saveOldTemperature(meshStruct);
}

void Albany::APFDiscretization::writeAnySolutionToFile(
      const double time_value)
{
  // Skip this write unless the proper interval has been reached.
  if (outputInterval++ % meshStruct->outputInterval)
    return;

  if (!meshOutput)
    return;

  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Write To File");

  double time_label = monotonicTimeLabel(time_value);
  int out_step = 0;

  if (mapT->getComm()->getRank()==0) {
    *out << "Albany::APFDiscretization::writeSolution: writing time " << time_value;
    if (time_label != time_value) *out << " with label " << time_label;
    *out << " to index " << out_step << " in file "
         << meshStruct->outputFileName << std::endl;
  }

  apf::Field* f;
  int dim = getNumDim();
  apf::FieldShape* fs = apf::getIPShape(dim, meshStruct->cubatureDegree);
  copyNodalDataToAPF(false);
  copyQPStatesToAPF(f,fs,false);
  meshOutput->writeFile(time_label);
  removeQPStatesFromAPF();
  removeNodalDataFromAPF();

  if ((continuationStep == meshStruct->restartWriteStep) &&
      (continuationStep != 0))
    writeRestartFile(time_label);

  continuationStep++;
}

void
Albany::APFDiscretization::writeRestartFile(const double time)
{
  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Write Restart");
  *out << "Albany::APFDiscretization::writeRestartFile: writing time "
    << time << std::endl;
  apf::Field* f;
  int dim = getNumDim();
  apf::FieldShape* fs = apf::getIPShape(dim, meshStruct->cubatureDegree);
  copyNodalDataToAPF(true);
  copyQPStatesToAPF(f,fs,true);
  apf::Mesh2* m = meshStruct->getMesh();
  std::ostringstream oss;
  oss << "restart_" << time << "_.smb";
  m->writeNative(oss.str().c_str());
  removeQPStatesFromAPF();
  removeNodalDataFromAPF();
}

void
Albany::APFDiscretization::writeMeshDebug (const std::string& filename) {
  apf::Field* f;
  apf::FieldShape* fs = apf::getIPShape(getNumDim(),
                                        meshStruct->cubatureDegree);
  copyQPStatesToAPF(f, fs, true);
  apf::writeVtkFiles(filename.c_str(), meshStruct->getMesh());
  removeQPStatesFromAPF();
}

double
Albany::APFDiscretization::monotonicTimeLabel(const double time)
{
  // If increasing, then all is good
  if (time > previous_time_label) {
    previous_time_label = time;
    return time;
  }
// Try absolute value
  double time_label = fabs(time);
  if (time_label > previous_time_label) {
    previous_time_label = time_label;
    return time_label;
  }

  // Try adding 1.0 to time
  if (time_label+1.0 > previous_time_label) {
    previous_time_label = time_label+1.0;
    return time_label+1.0;
  }

  // Otherwise, just add 1.0 to previous
  previous_time_label += 1.0;
  return previous_time_label;
}

void
Albany::APFDiscretization::setResidualFieldT(const Tpetra_Vector& residualT)
{
  Teuchos::ArrayRCP<const ST> data = residualT.get1dView();
  if (solLayout.getDerivNames(0).size() == 0) // dont have split fields
    this->setField(APFMeshStruct::residual_name,&(data[0]),/*overlapped=*/false);
  else
    this->setSplitFields(resNames, solLayout.getDerivSizes(0), &(data[0]), /*overlapped=*/false);

  meshStruct->residualInitialized = true;
}

#if 0 //defined(ALBANY_EPETRA)
void
Albany::APFDiscretization::setResidualField(const Epetra_Vector& residual)
{
  if (solLayout.getDerivNames(0).size() == 0)
    this->setField(APFMeshStruct::residual_name,&(residual[0]),/*overlapped=*/false);
  else
    this->setSplitFields(resNames, solLayout.getDerivSizes(0), &(residual[0]), /*overlapped=*/false);

  meshStruct->residualInitialized = true;
}
#endif

Teuchos::RCP<Tpetra_Vector>
Albany::APFDiscretization::getSolutionFieldT(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_Vector> solnT = Teuchos::rcp(
    new Tpetra_Vector(overlapped ? overlap_mapT : mapT));
  {
    Teuchos::ArrayRCP<ST> data = solnT->get1dViewNonConst();

    if (meshStruct->solutionInitialized) {
      if (solLayout.getDerivNames(0).size() == 0)
        this->getField(APFMeshStruct::solution_name[0], &(data[0]), overlapped);
      else
        this->getSplitFields(solLayout.getDerivNames(0), solLayout.getDerivSizes(0), &(data[0]), overlapped);
    }
    else if ( ! PCU_Comm_Self())
      *out <<__func__<<": uninit field" << std::endl;
  }
  return solnT;
}

Teuchos::RCP<Tpetra_MultiVector>
Albany::APFDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_MultiVector> solnT = Teuchos::rcp(
    new Tpetra_MultiVector(overlapped ? overlap_mapT : mapT,
      meshStruct->num_time_deriv + 1,
      /*zero-out=*/false));

  for(int i = 0; i <= meshStruct->num_time_deriv; ++i){

    Teuchos::RCP<Tpetra_Vector> colT = solnT->getVectorNonConst(i);
    Teuchos::ArrayRCP<ST> data = colT->get1dViewNonConst();

    if (meshStruct->solutionInitialized) {
      if (solLayout.getDerivNames(i).size() == 0)
        this->getField(APFMeshStruct::solution_name[i], &(data[0]), overlapped);
      else
        this->getSplitFields(solLayout.getDerivNames(i), solLayout.getDerivSizes(i), &(data[0]), overlapped);
    } else if ( ! PCU_Comm_Self()) {
      if (solLayout.getDerivNames(i).size() == 0) {
        *out <<__func__<<": uninit field "
             << APFMeshStruct::solution_name[i] << '\n';
      } else {
        *out <<__func__<<": uninit fields "
             << solLayout.getDerivNames(i) << '\n';
      }
    }
  }
  return solnT;
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<Epetra_Vector>
Albany::APFDiscretization::getSolutionField(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(
    new Epetra_Vector(overlapped ? *overlap_map : *map));

  if (meshStruct->solutionInitialized) {
    if (solLayout.getDerivNames(0).size() == 0)
      this->getField(APFMeshStruct::solution_name[0], &((*soln)[0]), overlapped);
    else
      this->getSplitFields(solLayout.getDerivNames(0), solLayout.getDerivSizes(0), &((*soln)[0]), overlapped);
  }
  else if ( ! PCU_Comm_Self())
    *out <<__func__<<": uninit field" << std::endl;

  return soln;
}
#endif

int Albany::APFDiscretization::nonzeroesPerRow(const int neq) const
{
  int numDim = getNumDim();

  /* DAI: this function should be revisited for overall correctness,
     especially in the case of higher-order fields */
  int estNonzeroesPerRow;
  switch (numDim) {
  case 0: estNonzeroesPerRow=1*neq; break;
  case 1: estNonzeroesPerRow=3*neq; break;
  case 2: estNonzeroesPerRow=9*neq; break;
  case 3: estNonzeroesPerRow=27*neq; break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "APFDiscretization:  Bad numDim"<< numDim);
  }
  return estNonzeroesPerRow;
}

static void offsetNumbering(
    apf::GlobalNumbering* n,
    apf::DynamicArray<apf::Node> const& nodes)
{
  const GO startIdx = 2147483647L;
  for (int i=0; i < nodes.getSize(); ++i)
  {
    GO oldIdx = apf::getNumber(n, nodes[i]);
    GO newIdx = startIdx + oldIdx;
    number(n, nodes[i], newIdx);
  }
}

void Albany::APFDiscretization::computeOwnedNodesAndUnknowns()
{
  apf::Mesh* m = meshStruct->getMesh();
  if (globalNumbering) apf::destroyGlobalNumbering(globalNumbering);
  globalNumbering = apf::makeGlobal(apf::numberOwnedNodes(m,"owned"));
  apf::DynamicArray<apf::Node> ownedNodes;
  apf::getNodes(globalNumbering,ownedNodes);
  if (meshStruct->useDOFOffsetHack)
    offsetNumbering(globalNumbering, ownedNodes);
  numOwnedNodes = ownedNodes.getSize();
  apf::synchronize(globalNumbering);
  Teuchos::Array<GO> indices(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; ++i)
    indices[i] = apf::getNumber(globalNumbering,ownedNodes[i]);
  node_mapT = Tpetra::createNonContigMap<LO, GO>(indices, commT);
  numGlobalNodes = node_mapT->getMaxAllGlobalIndex() + 1;
  if(Teuchos::nonnull(meshStruct->nodal_data_base))
    meshStruct->nodal_data_base->resizeLocalMap(indices, commT);
  indices.resize(numOwnedNodes*neq);
  for (int i=0; i < numOwnedNodes; ++i)
    for (int j=0; j < neq; ++j) {
      GO gid = apf::getNumber(globalNumbering,ownedNodes[i]);
      indices[getDOF(i,j)] = getDOF(gid,j);
    }
  mapT = Tpetra::createNonContigMap<LO, GO>(indices, commT);
#if defined(ALBANY_EPETRA)
  map = Teuchos::rcp(
    new Epetra_Map(-1, indices.size(), convert(indices)->getRawPtr(), 0,
                   *comm));
#endif
}

void Albany::APFDiscretization::computeOverlapNodesAndUnknowns()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Numbering* overlap = m->findNumbering("overlap");
  if (overlap) apf::destroyNumbering(overlap);
  overlap = apf::numberOverlapNodes(m,"overlap");
  apf::getNodes(overlap,nodes);
  numOverlapNodes = nodes.getSize();
  Teuchos::Array<GO> nodeIndices(numOverlapNodes);
  Teuchos::Array<GO> dofIndices(numOverlapNodes*neq);
  for (int i=0; i < numOverlapNodes; ++i) {
    GO global = apf::getNumber(globalNumbering,nodes[i]);
    nodeIndices[i] = global;
    for (int j=0; j < neq; ++j)
      dofIndices[getDOF(i,j)] = getDOF(global,j);
  }
  overlap_node_mapT = Tpetra::createNonContigMap<LO, GO>(nodeIndices, commT);
  overlap_mapT = Tpetra::createNonContigMap<LO, GO>(dofIndices, commT);
#if defined(ALBANY_EPETRA)
  overlap_map = Teuchos::rcp(
    new Epetra_Map(-1, dofIndices.size(), convert(dofIndices)->getRawPtr(), 0,
                   *comm));
#endif
  if(Teuchos::nonnull(meshStruct->nodal_data_base))
    meshStruct->nodal_data_base->resizeOverlapMap(nodeIndices, commT);
}

void Albany::APFDiscretization::computeGraphs()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::FieldShape* shape = m->getShape();
  int numDim = m->getDimension();
  std::vector<apf::MeshEntity*> cells;
  std::vector<int> n_nodes_in_elem;
  cells.reserve(m->count(numDim));
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* e;
  GO node_sum = 0;
  while ((e = m->iterate(it))){
    cells.push_back(e);
    int nnodes = apf::countElementNodes(shape,m->getType(e));
    n_nodes_in_elem.push_back(nnodes);
    node_sum += nnodes;
  }
  m->end(it);
  int nodes_per_element = std::ceil((double)node_sum / (double)cells.size());
  /* construct the overlap graph of all local DOFs as they
     are coupled by element-node connectivity */
  overlap_graphT = Teuchos::rcp(new Tpetra_CrsGraph(
                 overlap_mapT, neq*nodes_per_element));
#if defined(ALBANY_EPETRA)
  overlap_graph =
    Teuchos::rcp(new Epetra_CrsGraph(Copy, *overlap_map,
                                     neq*nodes_per_element, false));
#endif
  for (size_t i=0; i < cells.size(); ++i) {
    apf::NewArray<long> cellNodes;
    apf::getElementNumbers(globalNumbering,cells[i],cellNodes);
    for (int j=0; j < n_nodes_in_elem[i]; ++j) {
      for (int k=0; k < neq; ++k) {
        GO row = getDOF(cellNodes[j],k);
        for (int l=0; l < n_nodes_in_elem[i]; ++l) {
          for (int m=0; m < neq; ++m) {
            GO col = getDOF(cellNodes[l],m);
            Teuchos::ArrayView<GO> colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
#if defined(ALBANY_EPETRA)
            EpetraInt ecol = Teuchos::as<EpetraInt>(col);
            overlap_graph->InsertGlobalIndices(row,1,&ecol);
#endif
          }
        }
      }
    }
  }
  overlap_graphT->fillComplete();
#if defined(ALBANY_EPETRA)
  overlap_graph->FillComplete();
#endif

  // Create Owned graph by exporting overlap with known row map
  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));
#if defined(ALBANY_EPETRA)
  graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *map, nonzeroesPerRow(neq), false));
#endif

  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(
                                                       overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();

#if defined(ALBANY_EPETRA)
  Epetra_Export exporter(*overlap_map, *map);
  graph->Export(*overlap_graph, exporter, Insert);
  graph->FillComplete();
#endif
}

void Albany::APFDiscretization::computeWorksetInfo()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::FieldShape* shape = m->getShape();
  int numDim = m->getDimension();
  if (elementNumbering) apf::destroyGlobalNumbering(elementNumbering);
  elementNumbering = apf::makeGlobal(apf::numberElements(m,"element"));

/*
 * Note: Max workset size is given in input file, or set to a default in Albany_APFMeshStruct.cpp
 * The workset size is set in Albany_APFMeshStruct.cpp to be the maximum number in an element block if
 * the element block size < Max workset size.
 * STK bucket size is set to the workset size. We will "chunk" the elements into worksets here.
 */

  int worksetSize = meshStruct->worksetSize;

  buckets.clear();

  /* even with exponential growth, appending to a vector
   * of vectors may be expensive is std::move-style optimizations
   * aren't done.
   * we know there will be at least (#elements / worksetSize) worksets,
   * so reserve that much and avoid resizes until then. */
  buckets.reserve(m->count(numDim) / worksetSize + 1);

  /* likewise, wsEBNames is of a type which does not
   * resize exponentially, so during this procedure
   * we'll replace it with a std::vector */
  std::vector<std::string> wsEBNames_vec;
  wsEBNames_vec.reserve(m->count(numDim) / worksetSize + 1);

  std::map<apf::StkModel*, int> bucketMap;
  std::map<apf::StkModel*, int>::iterator buck_it;
  apf::StkModels& sets = meshStruct->getSets();
  int bucket_counter = 0;

  // iterate over all elements
  apf::MeshIterator* it = m->begin(numDim);
  apf::MeshEntity* element;
  while ((element = m->iterate(it)))
  {
    apf::ModelEntity* mr = m->toModel(element);
    apf::StkModel* block = sets.invMaps[numDim][mr];
    TEUCHOS_TEST_FOR_EXCEPTION(!block, std::logic_error,
        "No element block for model region " << m->getModelTag(mr)
        << " at " << __FILE__ << " +" << __LINE__ << '\n');
    // find the latest bucket being filled with elements for this block
    buck_it = bucketMap.find(block);
    if((buck_it == bucketMap.end()) ||  // this block hasn't been encountered yet
       (buckets[buck_it->second].size() >= worksetSize)){ // the current bucket for this block is "full"
      // Associate this elem_blk with the new bucket
      bucketMap[block] = bucket_counter;
      // start this new bucket off with the current element
      buckets.push_back(std::vector<apf::MeshEntity*>(1,element));
      // associate a bucket (workset) with an element block via a string
      wsEBNames_vec.push_back(block->stkName);
      bucket_counter++;
    }
    else { // put the element in the proper bucket
      buckets[buck_it->second].push_back(element);
    }
  }
  m->end(it);

  /* now copy the std::vector into the plain array */
  wsEBNames.resize(wsEBNames_vec.size());
  for (size_t i = 0; i < wsEBNames_vec.size(); ++i)
    wsEBNames[i] = wsEBNames_vec[i];

  int numBuckets = bucket_counter;

  wsPhysIndex.resize(numBuckets);

  if (meshStruct->allElementBlocksHaveSamePhysics)
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=0;
  else
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=meshStruct->ebNameToIndex[wsEBNames[i]];

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);
  latticeOrientation.resize(numBuckets);

  // Clear map if remeshing
  if(!elemGIDws.empty()) elemGIDws.clear();

  /* this block of code creates the wsElNodeEqID,
     wsElNodeID, and coords structures.
     These are (bucket, element, element_node, dof)-indexed
     structures to get numbers or coordinates */
  for (int b=0; b < numBuckets; b++) {

    std::vector<apf::MeshEntity*>& buck = buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // i is the element index within bucket b

    for (std::size_t i=0; i < buck.size(); i++) {

      // Traverse all the elements in this bucket
      element = buck[i];
      apf::Node node(element,0);

      // Now, save a map from element GID to workset on this PE
      elemGIDws[apf::getNumber(elementNumbering,node)].ws = b;

      // Now, save a map element GID to local id on this workset on this PE
      elemGIDws[apf::getNumber(elementNumbering,node)].LID = i;

      // get global node numbers
      apf::NewArray<long> nodeIDs;
      apf::getElementNumbers(globalNumbering,element,nodeIDs);

      int nodes_per_element = apf::countElementNodes(
          shape,m->getType(element));
      wsElNodeEqID[b][i].resize(nodes_per_element);
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes
      const int spdim = getNumDim();
      for (int j=0; j < nodes_per_element; j++) {
        const GO node_gid = nodeIDs[j];
        const LO node_lid = overlap_node_mapT->getLocalElement(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
            "PUMI: node_lid " << node_lid << " out of range\n");

        coords[b][i][j] = &coordinates[node_lid * spdim];
        wsElNodeEqID[b][i][j].resize(neq);
        wsElNodeID[b][i][j] = node_gid;

        for (std::size_t eq=0; eq < neq; eq++)
          wsElNodeEqID[b][i][j][eq] = getDOF(node_lid,eq);
      }
    }
  }

  // (Re-)allocate storage for element data
  //
  // For each state, create storage for the data for on processor elements
  // elemGIDws.size() is the number of elements on this processor ...
  // Note however that Intrepid2 will stride over numBuckets * worksetSize
  // so we must allocate enough storage for that

  std::size_t numElementsAccessed = numBuckets * worksetSize;

  for (std::size_t i=0; i<meshStruct->qpscalar_states.size(); i++)
      meshStruct->qpscalar_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<meshStruct->qpvector_states.size(); i++)
      meshStruct->qpvector_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<meshStruct->qptensor_states.size(); i++)
      meshStruct->qptensor_states[i]->reAllocateBuffer(numElementsAccessed);
  for (std::size_t i=0; i<meshStruct->scalarValue_states.size(); i++)
      // special case : need to store one double value that represents all the elements in the workset (time)
      // numBuckets are the number of worksets
      meshStruct->scalarValue_states[i]->reAllocateBuffer(numBuckets);
  for (std::size_t i=0; i<meshStruct->elemnodescalar_states.size(); ++i)
      meshStruct->elemnodescalar_states[i]->reAllocateBuffer(numElementsAccessed);

  // Pull out pointers to shards::Arrays for every bucket, for every state

  // Note that numBuckets is typically larger each time the mesh is adapted

  stateArrays.elemStateArrays.resize(numBuckets);

  for (std::size_t b=0; b < buckets.size(); b++) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    for (std::size_t i=0; i<meshStruct->qpscalar_states.size(); i++)
      stateArrays.elemStateArrays[b][meshStruct->qpscalar_states[i]->name] =
                 meshStruct->qpscalar_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<meshStruct->qpvector_states.size(); i++)
      stateArrays.elemStateArrays[b][meshStruct->qpvector_states[i]->name] =
                 meshStruct->qpvector_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<meshStruct->qptensor_states.size(); i++)
      stateArrays.elemStateArrays[b][meshStruct->qptensor_states[i]->name] =
                 meshStruct->qptensor_states[i]->getMDA(buck.size());
    for (std::size_t i=0; i<meshStruct->scalarValue_states.size(); i++)
      stateArrays.elemStateArrays[b][meshStruct->scalarValue_states[i]->name] =
                 meshStruct->scalarValue_states[i]->getMDA(1);
    for (std::size_t i=0; i<meshStruct->elemnodescalar_states.size(); ++i) {
      stateArrays.elemStateArrays[b][meshStruct->elemnodescalar_states[i]->name] =
                 meshStruct->elemnodescalar_states[i]->getMDA(buck.size());
    }
  }

// Process node data sets if present

  if(Teuchos::nonnull(meshStruct->nodal_data_base) &&
    meshStruct->nodal_data_base->isNodeDataPresent()) {

    std::vector< std::vector<apf::Node> > nbuckets; // bucket of nodes
    int numNodeBuckets =  (int)ceil((double)numOwnedNodes / (double)worksetSize);

    nbuckets.resize(numNodeBuckets);
    int node_bucket_counter = 0;
    int node_in_bucket = 0;

    // iterate over all nodes and save the owned ones into buckets
    for (size_t i=0; i < nodes.getSize(); ++i)
      if (m->isOwned(nodes[i].entity))
      {
        nbuckets[node_bucket_counter].push_back(nodes[i]);
        node_in_bucket++;
        if (node_in_bucket >= worksetSize) {
          ++node_bucket_counter;
          node_in_bucket = 0;
        }
      }

    Teuchos::RCP<Albany::NodeFieldContainer> node_states = meshStruct->nodal_data_base->getNodeContainer();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);

    // Loop over all the node field containers
    for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
                nfs != node_states->end(); ++nfs){
      Teuchos::RCP<Albany::AbstractPUMINodeFieldContainer> nodeContainer =
             Teuchos::rcp_dynamic_cast<Albany::AbstractPUMINodeFieldContainer>((*nfs).second);

      // resize the container to hold all the owned node's data
      nodeContainer->resize(node_mapT);

      // Now, loop over each workset to get a reference to each workset collection of nodes
      for (std::size_t b=0; b < nbuckets.size(); b++) {
         std::vector<apf::Node>& buck = nbuckets[b];
         stateArrays.nodeStateArrays[b][(*nfs).first] = nodeContainer->getMDA(buck);
      }
    }
  }
}

void Albany::APFDiscretization::computeNodeSets()
{
  // Make sure all the maps are allocated
  for (int i = 0; i < meshStruct->nsNames.size(); i++)
  { // Iterate over Node Sets
    std::string name = meshStruct->nsNames[i];
    nodeSets[name].resize(0);
    nodeSetCoords[name].resize(0);
    nodeset_node_coords[name].resize(0);
  }
  //grab the analysis model and mesh
  apf::StkModels& sets = meshStruct->getSets();
  apf::Mesh* m = meshStruct->getMesh();
  int mesh_dim = m->getDimension();
  //loop over mesh nodes
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    apf::MeshEntity* e = node.entity;
    if (!m->isOwned(e))
      continue;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(m, sets.invMaps[0], m->toModel(e), mset);
    if (mset.empty())
      continue;
    GO node_gid = apf::getNumber(globalNumbering,node);
    int node_lid = node_mapT->getLocalElement(node_gid);
    assert(node_lid >= 0);
    assert(node_lid < numOwnedNodes);
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* ns = *mit;
      std::string const& NS_name = ns->stkName;
      nodeSets[NS_name].push_back(std::vector<int>());
      std::vector<int>& dofLids = nodeSets[NS_name].back();
      std::vector<double>& ns_coords = nodeset_node_coords[NS_name];
      ns_coords.resize(ns_coords.size() + mesh_dim);
      double* node_coords = &ns_coords[ns_coords.size() - mesh_dim];
      nodeSetCoords[NS_name].push_back(node_coords);
      dofLids.resize(neq);
      for (std::size_t eq=0; eq < neq; eq++)
        dofLids[eq] = getDOF(node_lid, eq);
      double buf[3];
      apf::getComponents(m->getCoordinateField(), e, node.node, buf);
      for (int j = 0; j < mesh_dim; ++j) node_coords[j] = buf[j];
    }
  }
}

void Albany::APFDiscretization::computeSideSets()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::StkModels& sets = meshStruct->getSets();

  // need a sideset list per workset
  int num_buckets = wsEBNames.size();
  sideSets.clear();
  sideSets.resize(num_buckets);

  int d = m->getDimension();

  // loop over mesh sides
  apf::MeshIterator* it = m->begin(d - 1);
  apf::MeshEntity* side;
  while ((side = m->iterate(it))) {
    apf::ModelEntity* me = m->toModel(side);
    if (!sets.invMaps[d - 1].count(me))
      continue;
    //side is part of a side set
    apf::StkModel* sideSet = sets.invMaps[d - 1][me];
    std::string const& ss_name = sideSet->stkName;

    // get the elements adjacent to this side
    apf::Up side_elems;
    m->getUp(side, side_elems);

    // we are not yet considering non-manifold side sets !
    TEUCHOS_TEST_FOR_EXCEPTION(side_elems.n != 1, std::logic_error,
		 "PUMIDisc: cannot figure out side set topology for side set "<<ss_name<<std::endl);

    apf::MeshEntity* elem = side_elems.e[0];

    // fill in the data holder for a side struct

    Albany::SideStruct sstruct;

    sstruct.elem_GID = apf::getNumber(elementNumbering, apf::Node(elem, 0));
    int workset = elemGIDws[sstruct.elem_GID].ws; // workset ID that this element lives in
    sstruct.elem_LID = elemGIDws[sstruct.elem_GID].LID; // local element id in this workset
    sstruct.elem_ebIndex = meshStruct->ebNameToIndex[wsEBNames[workset]]; // element block that workset lives in
    sstruct.side_local_id = apf::getLocalSideId(m, elem, side);

    Albany::SideSetList& ssList = sideSets[workset]; // Get a ref to the side set map for this ws

    // Get an iterator to the correct sideset (if it exists)
    Albany::SideSetList::iterator sit = ssList.find(ss_name);

    if (sit != ssList.end()) // The sideset has already been created
      sit->second.push_back(sstruct); // Save this side to the vector that belongs to the name ss->first
    else { // Add the key ss_name to the map, and the side vector to that map
      std::vector<Albany::SideStruct> tmpSSVec;
      tmpSSVec.push_back(sstruct);
      ssList.insert(Albany::SideSetList::value_type(ss_name, tmpSSVec));
    }
  }
  m->end(it);
}

void Albany::APFDiscretization::copyQPScalarToAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e)
      for (std::size_t p=0; p < nqp; ++p)
        apf::setScalar(f,buck[e],p,ar(e,p));
  }
}

void Albany::APFDiscretization::copyQPVectorToAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  const int spdim = meshStruct->problemDim;
  apf::Vector3 v(0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < spdim; ++i)
          v[i] = ar(e,p,i);
        apf::setVector(f,buck[e],p,v);
      }
    }
  }
}

void Albany::APFDiscretization::copyQPTensorToAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  const int spdim = meshStruct->problemDim;
  apf::Matrix3x3 v(0,0,0,0,0,0,0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < spdim; ++i)
          for (std::size_t j=0; j < spdim; ++j)
            v[i][j] = ar(e,p,i,j);
        apf::setMatrix(f,buck[e],p,v);
      }
    }
  }
}

void Albany::APFDiscretization::copyQPStatesToAPF(
    apf::Field* f,
    apf::FieldShape* fs,
    bool copyAll)
{
  apf::Mesh2* m = meshStruct->getMesh();
  for (std::size_t i=0; i < meshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(meshStruct->qpscalar_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::SCALAR,fs);
    copyQPScalarToAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(meshStruct->qpvector_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::VECTOR,fs);
    copyQPVectorToAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(meshStruct->qptensor_states[i]);
    if (!copyAll && !state.output)
      continue;
    int nqp = state.dims[1];
    f = apf::createField(m,state.name.c_str(),apf::MATRIX,fs);
    copyQPTensorToAPF(nqp, state.name, f);
  }
  if (meshStruct->saveStabilizedStress)
    saveStabilizedStress();
}

void Albany::APFDiscretization::removeQPStatesFromAPF()
{
  apf::Mesh2* m = meshStruct->getMesh();
  for (std::size_t i=0; i < meshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(meshStruct->qpscalar_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < meshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(meshStruct->qpvector_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
  for (std::size_t i=0; i < meshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(meshStruct->qptensor_states[i]);
    apf::destroyField(m->findField(state.name.c_str()));
  }
}

void Albany::APFDiscretization::copyQPScalarFromAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  apf::Mesh2* m = meshStruct->getMesh();
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p = 0; p < nqp; ++p) {
        ar(e,p) = apf::getScalar(f,buck[e],p);
      }
    }
  }
}

void Albany::APFDiscretization::copyQPVectorFromAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  const int spdim = meshStruct->problemDim;
  apf::Mesh2* m = meshStruct->getMesh();
  apf::Vector3 v(0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getVector(f,buck[e],p,v);
        for (std::size_t i=0; i < spdim; ++i)
          ar(e,p,i) = v[i];
      }
    }
  }
}

void Albany::APFDiscretization::copyQPTensorFromAPF(
    unsigned nqp,
    std::string const& stateName,
    apf::Field* f)
{
  const int spdim = meshStruct->problemDim;
  apf::Mesh2* m = meshStruct->getMesh();
  apf::Matrix3x3 v(0,0,0,0,0,0,0,0,0);
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    Albany::MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getMatrix(f,buck[e],p,v);
        for (std::size_t i=0; i < spdim; ++i) {
          for (std::size_t j=0; j < spdim; ++j)
            ar(e,p,i,j) = v[i][j];
        }
      }
    }
  }
}

void Albany::APFDiscretization::copyQPStatesFromAPF()
{
  apf::Mesh2* m = meshStruct->getMesh();
  apf::Field* f;
  for (std::size_t i=0; i < meshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(meshStruct->qpscalar_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    if (f)
      copyQPScalarFromAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(meshStruct->qpvector_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    if (f)
      copyQPVectorFromAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(meshStruct->qptensor_states[i]);
    int nqp = state.dims[1];
    f = m->findField(state.name.c_str());
    if (f)
      copyQPTensorFromAPF(nqp, state.name, f);
  }
}

void Albany::APFDiscretization::
copyNodalDataToAPF (const bool copy_all) {
  if (meshStruct->nodal_data_base.is_null()) return;
  const Teuchos::RCP<Albany::NodeFieldContainer>
    node_states = meshStruct->nodal_data_base->getNodeContainer();
  apf::Mesh2* const m = meshStruct->getMesh();

  for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
       nfs != node_states->end(); ++nfs) {
    Teuchos::RCP<Albany::PUMINodeDataBase<RealType> >
      nd = Teuchos::rcp_dynamic_cast<Albany::PUMINodeDataBase<RealType>>(
        nfs->second);
    TEUCHOS_TEST_FOR_EXCEPTION(
      nd.is_null(), std::logic_error,
      "A node field container is not a PUMINodeDataBase");
    if ( ! copy_all && ! nd->output) continue;

    int value_type;
    switch (nd->ndims()) {
    case 0: value_type = apf::SCALAR; break;
    case 1: value_type = apf::VECTOR; break;
    case 2: value_type = apf::MATRIX; break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                 "dim is not in {1,2,3}");
    }
    apf::Field* f = meshStruct->createNodalField(nd->name.c_str(), value_type);
    if (!PCU_Comm_Self()) std::cerr << "setting nodal field " << nd->name;
    PCU_Barrier();
    setField(nd->name.c_str(), &nd->buffer[0], false, 0, false);
  }
}

void Albany::APFDiscretization::removeNodalDataFromAPF () {
  if (meshStruct->nodal_data_base.is_null()) return;
  const Teuchos::RCP<Albany::NodeFieldContainer>
    node_states = meshStruct->nodal_data_base->getNodeContainer();
  apf::Mesh2* m = meshStruct->getMesh();

  for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
       nfs != node_states->end(); ++nfs) {
    Teuchos::RCP<Albany::PUMINodeDataBase<RealType> >
      nd = Teuchos::rcp_dynamic_cast<Albany::PUMINodeDataBase<RealType>>(
        nfs->second);
    apf::destroyField(m->findField(nd->name.c_str()));
  }
}

void
Albany::APFDiscretization::
initTimeFromParamLib(Teuchos::RCP<ParamLib> paramLib) {
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    if (stateArrays.elemStateArrays[b].count("Time")) {
      TEUCHOS_TEST_FOR_EXCEPTION(
        !paramLib->isParameter("Time"), std::logic_error,
        "APF: Time is a state but not a parameter, cannot reinitialize it\n");
      Albany::MDArray& time = stateArrays.elemStateArrays[b]["Time"];
      time(0) = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    }
    if (stateArrays.elemStateArrays[b].count("Time_old")) {
      Albany::MDArray& oldTime = stateArrays.elemStateArrays[b]["Time_old"];
      oldTime(0) = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    }
  }
}

void
Albany::APFDiscretization::updateMesh(bool shouldTransferIPData) {
  updateMesh(shouldTransferIPData, Teuchos::null);
}

void
Albany::APFDiscretization::updateMesh(bool shouldTransferIPData,
    Teuchos::RCP<ParamLib> paramLib) {
  // This function is called both to initialize the mesh at the beginning of the simulation
  // and then each time the mesh is adapted (called from AAdapt_MeshAdapt_Def.hpp - afterAdapt())

  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Transfer to Albany");
  computeOwnedNodesAndUnknowns();
  computeOverlapNodesAndUnknowns();
  setupMLCoords();
  computeGraphs();
  getCoordinates(); //fill the coordinates array
  computeWorksetInfo();
  computeNodeSets();
  computeSideSets();

  // transfer of internal variables
  if (shouldTransferIPData)
    copyQPStatesFromAPF();

  // load the FELIX Data and tell the state manager to not initialize
  // these fields
  if (meshStruct->shouldLoadFELIXData)
    setFELIXData();

  // Tell the nodal data base that the graph changed. We don't create the graph
  // (as STKDiscretization does), but others might (such as
  // ProjectIPtoNodalField), so invalidate it.
  if (Teuchos::nonnull(meshStruct->nodal_data_base))
    meshStruct->nodal_data_base->updateNodalGraph(Teuchos::null);

  // Use the parameter library to re-initialize Time state arrays
  if (Teuchos::nonnull(paramLib))
    initTimeFromParamLib(paramLib);
}

void
Albany::APFDiscretization::attachQPData() {
  apf::Field* f;
  int order = meshStruct->cubatureDegree;
  int dim = meshStruct->getMesh()->getDimension();
  apf::FieldShape* fs = apf::getVoronoiShape(dim,order);
  assert(fs);
  copyQPStatesToAPF(f,fs);
}

void
Albany::APFDiscretization::detachQPData() {
  removeQPStatesFromAPF();
}

void Albany::APFDiscretization::releaseMesh () {
  if (globalNumbering) {
    apf::destroyGlobalNumbering(globalNumbering);
    globalNumbering = 0;
  }
  if (elementNumbering) {
    apf::destroyGlobalNumbering(elementNumbering);
    elementNumbering = 0;
  }
}

static apf::Field* interpolate(
    apf::Field* nf,
    int cubatureDegree,
    char const* name)
{
  assert(apf::getValueType(nf) == apf::SCALAR);
  apf::Mesh* m = apf::getMesh(nf);
  int dim = m->getDimension();
  apf::FieldShape* qpfs = apf::getIPShape(dim, cubatureDegree);
  apf::Field* ipf = apf::createField(m, name, apf::SCALAR, qpfs);
  apf::MeshIterator* it = m->begin(dim);
  apf::MeshEntity* e;
  while ((e = m->iterate(it))) {
    apf::Mesh::Type et = m->getType(e);
    apf::Element* fe = apf::createElement(nf, e);
    unsigned nqp = apf::countGaussPoints(et, cubatureDegree);
    for (unsigned i = 0; i < nqp; ++i) {
      apf::Vector3 xi;
      apf::getGaussPoint(et, cubatureDegree, i, xi);
      double val = apf::getScalar(fe, xi);
      apf::setScalar(ipf, e, i, val);
    }
    apf::destroyElement(fe);
  }
  m->end(it);
  return ipf;
}

static apf::Field* try_interpolate(
    apf::Mesh* m,
    char const* fromName,
    int cubatureDegree,
    char const* toName)
{
  apf::Field* nf = m->findField(fromName);
  if (!nf) {
    std::cout << "could not find " << fromName << " on nodes\n";
    return 0;
  } else {
    std::cout << "interpolating nodal " << fromName << " to QP " << toName << '\n';
    return interpolate(nf, cubatureDegree, toName);
  }
}

static void temperaturesToQP(
    apf::Mesh* m,
    int cubatureDegree)
{
  int o = cubatureDegree;
  if (!try_interpolate(m, "temp", o, "Temperature"))
    try_interpolate(m, Albany::APFMeshStruct::solution_name[0], o, "Temperature");
  if (!try_interpolate(m, "temp_old", o, "Temperature_old"))
    if (!try_interpolate(m, "temp", o, "Temperature_old"))
      try_interpolate(m, Albany::APFMeshStruct::solution_name[0], o, "Temperature_old");
}

/* LCM's ThermoMechanicalCoefficients evaluator
   relies on Temperature and Temperature_old
   to be initialized in the stateArrays as well
   as the solution vector.
   this hack will interpolate values from the
   solution vector "temp" to populate the
   stateArrays */

void
Albany::APFDiscretization::initTemperatureHack() {
  if (!meshStruct->useTemperatureHack)
    return;
  apf::Mesh* m = meshStruct->getMesh();
  temperaturesToQP(m, meshStruct->cubatureDegree);
  copyQPStatesFromAPF();
  apf::destroyField(m->findField("Temperature"));
  apf::destroyField(m->findField("Temperature_old"));
}
