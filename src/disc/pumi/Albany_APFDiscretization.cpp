//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_APFDiscretization.hpp"

#include "Albany_Utils.hpp"
#include "Albany_PUMIOutput.hpp"
#if defined(ALBANY_CONTACT)
#include "Albany_ContactManager.hpp"
#endif
#include "PHAL_AlbanyTraits.hpp"
#include <PHAL_Dimension.hpp>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <apfMesh.h>
#include <apfShape.h>
#include <PCU.h>

#include <string>
#include <iostream>
#include <fstream>
#include <limits>

namespace Albany {

APFDiscretization::
APFDiscretization(Teuchos::RCP<APFMeshStruct> meshStruct_,
                  const Teuchos::RCP<const Teuchos_Comm>& comm_,
                  const Teuchos::RCP<RigidBodyModes>& rigidBodyModes_)
 : out(Teuchos::VerboseObjectBase::getDefaultOStream())
 , previous_time_label(-1.0e32)
 , comm(comm_)
 , neq(meshStruct_->neq)
 , meshStruct(meshStruct_)
 , interleavedOrdering(meshStruct_->interleavedOrdering)
 , rigidBodyModes(rigidBodyModes_)
 , outputInterval(0)
 , continuationStep(0)
{
  // Nothing to do here
}

APFDiscretization::~APFDiscretization() {
  delete meshOutput;
}

void APFDiscretization::init()
{
  meshOutput = PUMIOutput::create(meshStruct, comm);
  globalNumbering = 0;
  elementNumbering = 0;

  // Initialize the mesh and all data structures for Albany
  initMesh();

  // layout[num deriv vectors][DOF_component]
  Teuchos::Array<Teuchos::Array<std::string> > layout = meshStruct->solVectorLayout;
  int number_of_solution_vecs = layout.size();
  solLayout.resize(number_of_solution_vecs);


  for (int i=0; i<layout[0].size(); i+=2) {
    std::string res_name = layout[0][i];
    res_name.append("Res");

    resNames.push_back(res_name);
  }

  for (int j=0; j < number_of_solution_vecs; j++) {
    unsigned int total_ndofs = 0;
    for (int i=0; i<layout[j].size(); i += 2) {
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
  if (resNames.size()) {
    for (int i=0; i<resNames.size(); ++i) {
      apf::zeroField(meshStruct->getMesh()->findField(resNames[i].c_str()));
    }
  } else {
    apf::zeroField(meshStruct->getMesh()->findField(APFMeshStruct::residual_name));
  }

  // set all of the restart fields here
  if (meshStruct->hasRestartSolution) {
    setRestartData();
  }
}

void APFDiscretization::printCoords() const
{
  int mesh_dim = meshStruct->getMesh()->getDimension();

  std::cout << "Processor " << PCU_Comm_Self() << " has " << coords.size()
            << " worksets." << std::endl;

  for (int ws=0; ws<coords.size(); ws++) {  //workset
    for (int e=0; e<coords[ws].size(); e++) { //cell
      for (int j=0; j<coords[ws][e].size(); j++) { //node
        for (int d=0; d<mesh_dim; d++) { //dim
          std::cout << "Coord for workset: " << ws << " element: " << e
                    << " node: " << j << " DOF: " << d
                    << " is: " << coords[ws][e][j][d] << std::endl;
  }}}}
}

const Teuchos::ArrayRCP<double>&
APFDiscretization::getCoordinates() const
{
  const int spdim = getNumDim();
  coordinates.resize(spdim * numOverlapNodes);
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();
  for (size_t i = 0; i < overlapNodes.getSize(); ++i) {
    if (spdim == 3)
      apf::getComponents(f, overlapNodes[i].entity, overlapNodes[i].node, &coordinates[3*i]);
    else {
      double buf[3];
      apf::getComponents(f, overlapNodes[i].entity, overlapNodes[i].node, buf);
      double* const c = &coordinates[spdim*i];
      for (int j = 0; j < spdim; ++j) c[j] = buf[j];
    }
  }
  return coordinates;
}

void APFDiscretization::
setCoordinates(const Teuchos::ArrayRCP<const double>& c)
{
  const int spdim = getNumDim();
  double buf[3] = {0};
  apf::Field* f = meshStruct->getMesh()->getCoordinateField();
  for (size_t i = 0; i < overlapNodes.getSize(); ++i) {
    if (spdim == 3) {
      apf::setComponents(f, overlapNodes[i].entity, overlapNodes[i].node, &c[spdim*i]);
    } else {
      const double* const cp = &c[spdim*i];
      for (int j = 0; j < spdim; ++j) buf[j] = cp[j];
      apf::setComponents(f, overlapNodes[i].entity, overlapNodes[i].node, buf);
    }
  }
}

void APFDiscretization::
setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& ircm)
{
  rcm = ircm;
}

/* DAI: this function also has to change for high-order fields */
void APFDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) { return; }
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed()) { return; }

  // get mesh dimension and part handle
  const int mesh_dim = getNumDim();
  coordMV = Thyra::createMembers(m_node_vs,mesh_dim);

  apf::Field* f = meshStruct->getMesh()->getCoordinateField();

  auto coords_data = getNonconstLocalData(coordMV);
  for (std::size_t i = 0; i < ownedNodes.getSize(); ++i) {
    double lcoords[3];
    apf::getComponents(f, ownedNodes[i].entity, ownedNodes[i].node, lcoords);
    for (int j=0; j<mesh_dim; ++j) {
      coords_data[j][i] = lcoords[j];
    }
  }

  if (meshStruct->useNullspaceTranslationOnly) {
    rigidBodyModes->setCoordinates(coordMV);
  } else {
    rigidBodyModes->setCoordinatesAndNullspace(coordMV, m_vs);
  }
}

inline int albanyCountComponents (const int problem_dim, const int pumi_value_type) {
  switch (pumi_value_type) {
    case apf::SCALAR: return 1;
    case apf::VECTOR: return problem_dim;
    case apf::MATRIX: return problem_dim * problem_dim;
    default: assert(0); return -1;
  }
  TEUCHOS_UNREACHABLE_RETURN (-1);
}

void APFDiscretization::
setField(const char* name, const ST* data, bool overlapped, int offset, bool neq_sized)
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* f = m->findField(name);
  ALBANY_ASSERT(f, "\nExpected field " << name << " doesn't exist!\n");

  const int problem_dim = meshStruct->problemDim;
  double data_buf[9] = {0};
  const int pumi_value_type = apf::getValueType(f);
  const int albany_nc = albanyCountComponents(problem_dim, pumi_value_type);
  const int total_comps = (neq_sized ? neq : albany_nc);

  // the simple front-packing of components below would not
  // be sufficient to deal with incoming 2x2 tensors, so assert
  // that we are passing data straight through if dealing with a tensor
  if (pumi_value_type == apf::MATRIX) {
    ALBANY_ASSERT(albany_nc == 9, "APFDiscretization::setField: "
       "field " << name << " is apf::MATRIX, but albany_nc = " << albany_nc);
  }

  apf::DynamicArray<apf::Node> const& nodes = overlapped ? overlapNodes : ownedNodes;
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    const int first_dof = getDOF(i, offset, total_comps);
    const double* datap = data + first_dof;
    for (int j = 0; j < albany_nc; ++j) {
      data_buf[j] = datap[j];
    }
    apf::setComponents(f, node.entity, node.node, data_buf);
  }

  if (!overlapped) {
    apf::synchronize(f);
  }
}

void APFDiscretization::
setSplitFields(const Teuchos::Array<std::string>& names,
               const Teuchos::Array<int>& sizes,
               const ST* data, bool overlapped)
{
  int offset = 0;
  for (int i=0; i < names.size(); ++i) {
    this->setField(names[i].c_str(), data, overlapped, offset);
    offset += sizes[i];
  }
}

void APFDiscretization::
getField(const char* name, ST* data, bool overlapped, int offset, bool neq_sized) const
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* f = m->findField(name);
  ALBANY_ASSERT(f, "\nExpected field " << name << " doesn't exist!\n");
  const int problem_dim = meshStruct->problemDim;
  const int pumi_value_type = apf::getValueType(f);
  const int albany_nc = albanyCountComponents(problem_dim, pumi_value_type);
  assert(albany_nc <= 3);
  const int total_comps = (neq_sized ? neq : albany_nc);
  apf::DynamicArray<apf::Node> const& nodes = overlapped ? overlapNodes : ownedNodes;
  for (size_t i = 0; i < nodes.getSize(); ++i) {
    apf::Node node = nodes[i];
    const int first_dof = getDOF(i, offset, total_comps);
    double buf[3];
    apf::getComponents(f, node.entity, node.node, buf);
    for (int j = 0; j < albany_nc; ++j) {
      data[first_dof + j] = buf[j];
    }
  }
}

void APFDiscretization::
getSplitFields(const Teuchos::Array<std::string>& names,
               const Teuchos::Array<int>& sizes, ST* data,
               bool overlapped) const
{
  int offset = 0;
  for (int i=0; i < names.size(); ++i) {
    this->getField(names[i].c_str(), data, overlapped, offset);
    offset += sizes[i];
  }
}

void APFDiscretization::
reNameExodusOutput(const std::string& str)
{
  if (meshOutput) {
    meshOutput->setFileName(str);
  }
}

static void saveOldTemperature(Teuchos::RCP<APFMeshStruct> meshStruct)
{
  if (!meshStruct->useTemperatureHack) {
    return;
  }

  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* t = m->findField("temp");
  if (!t) {
    t = m->findField(APFMeshStruct::solution_name[0]);
  }
  assert(t);
  apf::Field* told = m->findField("temp_old");
  if (!told) {
    told = meshStruct->createNodalField("temp_old", apf::SCALAR);
  }
  assert(told);
  std::cout << "copying nodal " << apf::getName(t)
            << " to nodal " << apf::getName(told) << '\n';
  apf::copyData(told, t);
}

void APFDiscretization::
writeAnySolutionToMeshDatabase (const ST* soln, const int index, const bool overlapped)
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

void APFDiscretization::writeAnySolutionToFile(const double time_value)
{
  // Skip this write unless the proper interval has been reached.
  if (outputInterval++ % meshStruct->outputInterval) {
    return;
  }

  if (!meshOutput) {
    return;
  }

  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Write To File");

  double time_label = monotonicTimeLabel(time_value);
  int out_step = 0;

  if (comm->getRank()==0) {
    *out << "APFDiscretization::writeSolution: writing time " << time_value;
    if (time_label != time_value) {
      *out << " with label " << time_label;
    }
    *out << " to index " << out_step << " in file "
         << meshStruct->outputFileName << std::endl;
  }

  int dim = getNumDim();
  apf::FieldShape* fs = apf::getIPShape(dim, meshStruct->cubatureDegree);
  copyNodalDataToAPF(false);
  copyQPStatesToAPF(fs,false);
  meshOutput->writeFile(time_label);
  removeQPStatesFromAPF();
  removeNodalDataFromAPF();

  if ((continuationStep == meshStruct->restartWriteStep) &&
      (continuationStep != 0)) {
    writeRestartFile(time_label);
  }

  continuationStep++;
}

void APFDiscretization::writeRestartFile(const double time)
{
  TEUCHOS_FUNC_TIME_MONITOR("AlbanyAdapt: Write Restart");
  *out << "APFDiscretization::writeRestartFile: writing time "
       << time << std::endl;
  int dim = getNumDim();
  apf::FieldShape* fs = apf::getIPShape(dim, meshStruct->cubatureDegree);
  copyNodalDataToAPF(true);
  copyQPStatesToAPF(fs,true);
  apf::Mesh2* m = meshStruct->getMesh();
  std::ostringstream oss;
  oss << "restart_" << time << "_.smb";
  m->writeNative(oss.str().c_str());
  removeQPStatesFromAPF();
  removeNodalDataFromAPF();
}

void APFDiscretization::writeMeshDebug (const std::string& filename) {
  apf::FieldShape* fs = apf::getIPShape(getNumDim(),
                                        meshStruct->cubatureDegree);
  copyQPStatesToAPF(fs, true);
  apf::writeVtkFiles(filename.c_str(), meshStruct->getMesh());
  removeQPStatesFromAPF();
}

double APFDiscretization::monotonicTimeLabel(const double time)
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

#if defined(ALBANY_LCM)
void APFDiscretization::setResidualField(const Thyra_Vector& residual)
{
  Teuchos::ArrayRCP<const ST> data = getLocalData(residual);
  if (solLayout.getDerivNames(0).size() == 0) {
    // dont have split fields
    this->setField(APFMeshStruct::residual_name,data.getRawPtr(),/*overlapped=*/false);
  } else {
    this->setSplitFields(resNames, solLayout.getDerivSizes(0), data.getRawPtr(), /*overlapped=*/false);
  }
  meshStruct->residualInitialized = true;
}
#endif

Teuchos::RCP<Thyra_Vector>
APFDiscretization::getSolutionField(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  const auto& vs = overlapped ? m_overlap_vs : m_vs;
  Teuchos::RCP<Thyra_Vector> solution = Thyra::createMember(vs);
  Teuchos::ArrayRCP<ST> data = getNonconstLocalData(solution);

  if (meshStruct->solutionInitialized) {
    if (solLayout.getDerivNames(0).size() == 0) {
      this->getField(APFMeshStruct::solution_name[0],
                     data.getRawPtr(),
                     overlapped);
    } else {
      this->getSplitFields(solLayout.getDerivNames(0),
                           solLayout.getDerivSizes(0),
                           data.getRawPtr(),
                           overlapped);
    }
  } else if ( ! PCU_Comm_Self()) {
    *out <<__func__<<": uninit field" << std::endl;
  }

  return solution;
}

Teuchos::RCP<Thyra_MultiVector>
APFDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  const auto& vs = overlapped ? m_overlap_vs : m_vs;
  Teuchos::RCP<Thyra_MultiVector> solution = Thyra::createMembers(vs,meshStruct->num_time_deriv+1);

  for(int i=0; i<=meshStruct->num_time_deriv; ++i) {
    auto col = solution->col(i);
    Teuchos::ArrayRCP<ST> data = getNonconstLocalData(col);

    if (meshStruct->solutionInitialized) {
      if (solLayout.getDerivNames(i).size() == 0) {
        this->getField(APFMeshStruct::solution_name[i],
                       data.getRawPtr(),
                       overlapped);
      } else {
        this->getSplitFields(solLayout.getDerivNames(i),
                             solLayout.getDerivSizes(i),
                             data.getRawPtr(),
                             overlapped);
      }
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
  return solution;
}

void APFDiscretization::
writeSolution (const Thyra_Vector& solution,
               const double time_value,
               const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = getLocalData(solution);
  writeAnySolutionToMeshDatabase(data.getRawPtr(), 0, overlapped);
  writeAnySolutionToFile(time_value);
}

void APFDiscretization::
writeSolution (const Thyra_Vector& solution,
               const Thyra_Vector& solution_dot,
               const double time_value,
              const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data     = getLocalData(solution);
  Teuchos::ArrayRCP<const ST> data_dot = getLocalData(solution_dot);
  writeAnySolutionToMeshDatabase(data.getRawPtr(),     0, overlapped);
  writeAnySolutionToMeshDatabase(data_dot.getRawPtr(), 0, overlapped);
  writeAnySolutionToFile(time_value);
}

void APFDiscretization::
writeSolution (const Thyra_Vector& solution,
               const Thyra_Vector& solution_dot, 
               const Thyra_Vector& solution_dotdot,
               const double time_value,
               const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data        = getLocalData(solution);
  Teuchos::ArrayRCP<const ST> data_dot    = getLocalData(solution_dot);
  Teuchos::ArrayRCP<const ST> data_dotdot = getLocalData(solution_dotdot);
  writeAnySolutionToMeshDatabase(data.getRawPtr(),        0, overlapped);
  writeAnySolutionToMeshDatabase(data_dot.getRawPtr(),    0, overlapped);
  writeAnySolutionToMeshDatabase(data_dotdot.getRawPtr(), 0, overlapped);
  writeAnySolutionToFile(time_value);
}

void APFDiscretization::
writeSolutionMV (const Thyra_MultiVector& solution,
                 const double time_value,
                 const bool overlapped)
{
  for(int i=0; i<=meshStruct->num_time_deriv; ++i) {
    auto col = solution.col(i);
    Teuchos::ArrayRCP<const ST> data = getLocalData(col);
    writeAnySolutionToMeshDatabase(data.getRawPtr(), i, overlapped);
  }

  writeAnySolutionToFile(time_value);
}

void APFDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const double /* time_value */,
                             const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data = getLocalData(solution);
  writeAnySolutionToMeshDatabase(data.getRawPtr(), 0, overlapped);
}

void APFDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const Thyra_Vector& solution_dot,
                             const double /* time_value */,
                             const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data        = getLocalData(solution);
  Teuchos::ArrayRCP<const ST> data_dot    = getLocalData(solution_dot);
  writeAnySolutionToMeshDatabase(data.getRawPtr(),     0, overlapped);
  writeAnySolutionToMeshDatabase(data_dot.getRawPtr(), 0, overlapped);
}

void APFDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const Thyra_Vector& solution_dot, 
                             const Thyra_Vector& solution_dotdot,
                             const double /* time_value */,
                             const bool overlapped)
{
  Teuchos::ArrayRCP<const ST> data        = getLocalData(solution);
  Teuchos::ArrayRCP<const ST> data_dot    = getLocalData(solution_dot);
  Teuchos::ArrayRCP<const ST> data_dotdot = getLocalData(solution_dotdot);
  writeAnySolutionToMeshDatabase(data.getRawPtr(),        0, overlapped);
  writeAnySolutionToMeshDatabase(data_dot.getRawPtr(),    0, overlapped);
  writeAnySolutionToMeshDatabase(data_dotdot.getRawPtr(), 0, overlapped);
}

void APFDiscretization::
writeSolutionMVToMeshDatabase (const Thyra_MultiVector& solution,
                               const double /* time_value */,
                               const bool overlapped)
{
  for(int i = 0; i <= meshStruct->num_time_deriv; i++){
    auto col = solution.col(i);
    Teuchos::ArrayRCP<const ST> data = getLocalData(col);
    writeAnySolutionToMeshDatabase(data.getRawPtr(), i, overlapped);
  }
}

void APFDiscretization::
writeSolutionToFile (const Thyra_Vector& solution,
                     const double time_value,
                     const bool /* overlapped */)
{
  // LB: This method does...nothing?!?!
  Teuchos::ArrayRCP<const ST> data = getLocalData(solution);
  writeAnySolutionToFile(time_value);
}

void APFDiscretization::
writeSolutionMVToFile (const Thyra_MultiVector& solution,
                       const double time_value,
                       const bool /* overlapped */)
{
  for(int i=0; i<=meshStruct->num_time_deriv; ++i) {
    auto col = solution.col(i);
    // LB: how is this writing the solution to file?!?!?
    Teuchos::ArrayRCP<const ST> data = getLocalData(col);
    writeAnySolutionToFile(time_value);
  }
}

int APFDiscretization::nonzeroesPerRow(const int num_eq) const
{
  int numDim = getNumDim();

  /* DAI: this function should be revisited for overall correctness,
     especially in the case of higher-order fields */
  int estNonzeroesPerRow;
  switch (numDim) {
    case 0: estNonzeroesPerRow=1*num_eq; break;
    case 1: estNonzeroesPerRow=3*num_eq; break;
    case 2: estNonzeroesPerRow=9*num_eq; break;
    case 3: estNonzeroesPerRow=27*num_eq; break;
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
  for (auto& node : nodes) {
    const GO oldIdx = apf::getNumber(n, node);
    const GO newIdx = startIdx + oldIdx;
    apf::number(n, node, newIdx);
  }
}

void APFDiscretization::computeOwnedNodesAndUnknowns()
{
  apf::Mesh* m = meshStruct->getMesh();
  assert(!globalNumbering);
  globalNumbering = apf::makeGlobal(apf::numberOwnedNodes(m,"owned"));
  apf::getNodes(globalNumbering,ownedNodes);
  if (meshStruct->useDOFOffsetHack) {
    offsetNumbering(globalNumbering, ownedNodes);
  }
  numOwnedNodes = ownedNodes.getSize();
  apf::synchronize(globalNumbering);
  Teuchos::Array<GO> indices(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; ++i) {
    indices[i] = apf::getNumber(globalNumbering,ownedNodes[i]);
  }
  m_node_vs = createVectorSpace(comm,indices);
  if(Teuchos::nonnull(meshStruct->nodal_data_base)) {
    meshStruct->nodal_data_base->replaceOwnedVectorSpace(indices, comm);
  }
  indices.resize(numOwnedNodes*neq);
  for (int i=0; i < numOwnedNodes; ++i) {
    for (unsigned int j=0; j < neq; ++j) {
      GO gid = apf::getNumber(globalNumbering,ownedNodes[i]);
      indices[getDOF(i,j)] = getDOF(gid,j);
    }
  }
  m_vs = createVectorSpace(comm,indices);
}

void APFDiscretization::computeOverlapNodesAndUnknowns()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Numbering* overlap = m->findNumbering("overlap");
  if (overlap) {
    apf::destroyNumbering(overlap);
  }

  overlap = apf::numberOverlapNodes(m,"overlap");
  apf::getNodes(overlap,overlapNodes);
  numOverlapNodes = overlapNodes.getSize();
  Teuchos::Array<GO> nodeIndices(numOverlapNodes);
  Teuchos::Array<GO> dofIndices(numOverlapNodes*neq);
  for (int i=0; i < numOverlapNodes; ++i) {
    GO global = apf::getNumber(globalNumbering,overlapNodes[i]);
    nodeIndices[i] = global;
    for (unsigned int j=0; j < neq; ++j)
      dofIndices[getDOF(i,j)] = getDOF(global,j);
  }
  m_overlap_node_vs = createVectorSpace(comm,nodeIndices);
  m_overlap_vs = createVectorSpace(comm,dofIndices);
  if(Teuchos::nonnull(meshStruct->nodal_data_base)) {
    meshStruct->nodal_data_base->replaceOverlapVectorSpace(nodeIndices, comm);
  }
}

void APFDiscretization::computeGraphs()
{
  apf::Mesh* mesh = meshStruct->getMesh();
  apf::FieldShape* shape = mesh->getShape();
  int numDim = mesh->getDimension();
  std::vector<apf::MeshEntity*> cells;
  std::vector<int> n_nodes_in_elem;
  cells.reserve(mesh->count(numDim));
  apf::MeshIterator* it = mesh->begin(numDim);
  apf::MeshEntity* e;
  GO node_sum = 0;
  while ((e = mesh->iterate(it))) {
    cells.push_back(e);
    int nnodes = apf::countElementNodes(shape,mesh->getType(e));
    n_nodes_in_elem.push_back(nnodes);
    node_sum += nnodes;
  }
  mesh->end(it);
  int nodes_per_element = std::ceil((double)node_sum / (double)cells.size());
  /* construct the overlap crs matrix factory of all local DOFs as they
     are coupled by element-node connectivity */
  m_overlap_jac_factory = Teuchos::rcp( new ThyraCrsMatrixFactory(m_overlap_vs,m_overlap_vs,neq*nodes_per_element) );

  for (size_t i=0; i < cells.size(); ++i) {
    apf::NewArray<long> cellNodes;
    apf::getElementNumbers(globalNumbering,cells[i],cellNodes);
    for (int j=0; j < n_nodes_in_elem[i]; ++j) {
      for (unsigned int k=0; k < neq; ++k) {
        GO row = getDOF(cellNodes[j],k);
        for (int l=0; l < n_nodes_in_elem[i]; ++l) {
          for (unsigned int m=0; m < neq; ++m) {
            GO col = getDOF(cellNodes[l],m);
            auto colAV = Teuchos::arrayView(&col, 1);
            m_overlap_jac_factory->insertGlobalIndices(row, colAV);
  }}}}}
  m_overlap_jac_factory->fillComplete();

  // Create Owned crs matrix factory by exporting overlap with owned vs
  m_jac_factory = Teuchos::rcp( new ThyraCrsMatrixFactory(m_vs,m_vs,m_overlap_jac_factory) );
}

void APFDiscretization::computeWorksetInfo()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::FieldShape* shape = m->getShape();
  int numDim = m->getDimension();
  assert(!elementNumbering);
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
  while ((element = m->iterate(it))) {
    apf::ModelEntity* mr = m->toModel(element);
    apf::StkModel* block = sets.invMaps[numDim][mr];
    TEUCHOS_TEST_FOR_EXCEPTION(!block, std::logic_error,
        "No element block for model region " << m->getModelTag(mr)
        << " at " << __FILE__ << " +" << __LINE__ << '\n');
    // find the latest bucket being filled with elements for this block
    buck_it = bucketMap.find(block);
    if((buck_it == bucketMap.end()) ||  // this block hasn't been encountered yet
       (buckets[buck_it->second].size() >= static_cast<size_t>(worksetSize))){ // the current bucket for this block is "full"
      // Associate this elem_blk with the new bucket
      bucketMap[block] = bucket_counter;
      // start this new bucket off with the current element
      buckets.push_back(std::vector<apf::MeshEntity*>(1,element));
      // associate a bucket (workset) with an element block via a string
      wsEBNames_vec.push_back(block->stkName);
      bucket_counter++;
    } else { // put the element in the proper bucket
      buckets[buck_it->second].push_back(element);
    }
  }
  m->end(it);

  /* now copy the std::vector into the plain array */
  wsEBNames.resize(wsEBNames_vec.size());
  for (size_t i = 0; i < wsEBNames_vec.size(); ++i) {
    wsEBNames[i] = wsEBNames_vec[i];
  }

  int numBuckets = bucket_counter;

  wsPhysIndex.resize(numBuckets);

  if (meshStruct->allElementBlocksHaveSamePhysics) {
    for (int i=0; i<numBuckets; i++) {
      wsPhysIndex[i]=0;
    }
  } else {
    for (int i=0; i<numBuckets; i++) {
      wsPhysIndex[i]=meshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[i]];
    }
  }

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);
  latticeOrientation.resize(numBuckets);
#if defined(ALBANY_LCM)
  boundary_indicator.resize(numBuckets);
#endif

  // Clear map if remeshing
  if(!elemGIDws.empty()) {
    elemGIDws.clear();
  }

  /* this block of code creates the wsElNodeEqID,
     wsElNodeID, and coords structures.
     These are (bucket, element, element_node, dof)-indexed
     structures to get numbers or coordinates */
  for (int b=0; b < numBuckets; b++) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // Set size of Kokkos views
    // Note: Assumes nodes_per_element is the same across all elements in a workset
    {
      const int buckSize = buck.size();
      element = buck[0];
      const int nodes_per_element = apf::countElementNodes(
          shape,m->getType(element));
      wsElNodeEqID[b] = WorksetConn("wsElNodeEqID", buckSize, nodes_per_element, neq);
    }

    // i is the element index within bucket b
    for (std::size_t i=0; i < buck.size(); i++) {

      // Traverse all the elements in this bucket
      element = buck[i];
      apf::Node node(element,0);

      GO elem_gid = apf::getNumber(elementNumbering,node);
      // Now, save a map from element GID to workset on this PE
      elemGIDws[elem_gid].ws = b;

      // Now, save a map element GID to local id on this workset on this PE
      elemGIDws[elem_gid].LID = i;

      // get global node numbers
      apf::NewArray<long> nodeIDs;
      apf::getElementNumbers(globalNumbering,element,nodeIDs);

      int nodes_per_element = apf::countElementNodes(
          shape,m->getType(element));
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      // loop over local nodes
      const int spdim = getNumDim();
      for (int j=0; j < nodes_per_element; j++) {
        const GO node_gid = nodeIDs[j];
        const LO node_lid = getLocalElement(m_overlap_node_vs,node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
            "PUMI: node_lid " << node_lid << " out of range\n");

        coords[b][i][j] = &coordinates[node_lid * spdim];
        wsElNodeID[b][i][j] = node_gid;

        for (std::size_t eq=0; eq < neq; eq++)
          wsElNodeEqID[b](i,j,eq) = getDOF(node_lid,eq);
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

  for (std::size_t i=0; i<meshStruct->qpscalar_states.size(); i++) {
      meshStruct->qpscalar_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<meshStruct->qpvector_states.size(); i++) {
      meshStruct->qpvector_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<meshStruct->qptensor_states.size(); i++) {
      meshStruct->qptensor_states[i]->reAllocateBuffer(numElementsAccessed);
  }
  for (std::size_t i=0; i<meshStruct->scalarValue_states.size(); i++) {
      // special case : need to store one double value that represents all the elements in the workset (time)
      // numBuckets are the number of worksets
      meshStruct->scalarValue_states[i]->reAllocateBuffer(numBuckets);
  }
  for (std::size_t i=0; i<meshStruct->elemnodescalar_states.size(); ++i) {
      meshStruct->elemnodescalar_states[i]->reAllocateBuffer(numElementsAccessed);
  }

  // Pull out pointers to shards::Arrays for every bucket, for every state

  // Note that numBuckets is typically larger each time the mesh is adapted

  stateArrays.elemStateArrays.resize(numBuckets);

  for (std::size_t b=0; b < buckets.size(); b++) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    for (std::size_t i=0; i<meshStruct->qpscalar_states.size(); i++) {
      stateArrays.elemStateArrays[b][meshStruct->qpscalar_states[i]->name] =
                 meshStruct->qpscalar_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<meshStruct->qpvector_states.size(); i++) {
      stateArrays.elemStateArrays[b][meshStruct->qpvector_states[i]->name] =
                 meshStruct->qpvector_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<meshStruct->qptensor_states.size(); i++) {
      stateArrays.elemStateArrays[b][meshStruct->qptensor_states[i]->name] =
                 meshStruct->qptensor_states[i]->getMDA(buck.size());
    }
    for (std::size_t i=0; i<meshStruct->scalarValue_states.size(); i++) {
      stateArrays.elemStateArrays[b][meshStruct->scalarValue_states[i]->name] =
                 meshStruct->scalarValue_states[i]->getMDA(1);
    }
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
    for (size_t i=0; i < overlapNodes.getSize(); ++i) {
      if (m->isOwned(overlapNodes[i].entity)) {
        nbuckets[node_bucket_counter].push_back(overlapNodes[i]);
        node_in_bucket++;
        if (node_in_bucket >= worksetSize) {
          ++node_bucket_counter;
          node_in_bucket = 0;
        }
      }
    }

    Teuchos::RCP<NodeFieldContainer> node_states = meshStruct->nodal_data_base->getNodeContainer();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);

    // Loop over all the node field containers
    for (NodeFieldContainer::iterator nfs = node_states->begin();
                nfs != node_states->end(); ++nfs){
      Teuchos::RCP<AbstractPUMINodeFieldContainer> nodeContainer =
             Teuchos::rcp_dynamic_cast<AbstractPUMINodeFieldContainer>((*nfs).second);

      // resize the container to hold all the owned node's data
      nodeContainer->resize(m_node_vs);

      // Now, loop over each workset to get a reference to each workset collection of nodes
      for (std::size_t b=0; b < nbuckets.size(); b++) {
         std::vector<apf::Node>& buck = nbuckets[b];
         stateArrays.nodeStateArrays[b][(*nfs).first] = nodeContainer->getMDA(buck);
      }
    }
  }
}

void APFDiscretization::
forEachNodeSetNode(std::function<void(size_t, apf::StkModel*)> fn)
{
  //grab the analysis model and mesh
  apf::StkModels& sets = meshStruct->getSets();
  apf::Mesh* m = meshStruct->getMesh();
  //loop over owned mesh nodes
  for (size_t i = 0; i < ownedNodes.getSize(); ++i) {
    apf::Node node = ownedNodes[i];
    apf::MeshEntity* e = node.entity;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(m, sets.invMaps[0], m->toModel(e), mset);
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* node_set = *mit;
      fn(i, node_set);
    }
  }
}

void APFDiscretization::computeNodeSets()
{
  // Make sure all the maps are allocated
  for (unsigned int i = 0; i < meshStruct->nsNames.size(); i++) {
    // Iterate over Node Sets
    std::string const& name = meshStruct->nsNames[i];
    nodeSets[name].resize(0);
    nodeSetCoords[name].resize(0);
    nodeset_node_coords[name].resize(0);
  }
  std::map<std::string, int> nodeSetSizes;
  auto count_fn = [&](size_t, apf::StkModel* node_set)
  {
    std::string const& NS_name = node_set->stkName;
    ++(nodeSetSizes[NS_name]);
  };
  forEachNodeSetNode(count_fn);
  apf::Mesh* m = meshStruct->getMesh();
  int mesh_dim = m->getDimension();
  for (unsigned int i = 0; i < meshStruct->nsNames.size(); i++) {
    std::string const& name = meshStruct->nsNames[i];
    nodeset_node_coords[name].resize(nodeSetSizes[name] * mesh_dim);
    nodeSetSizes[name] = 0;
  }
  auto fill_fn = [&](size_t owned_i, apf::StkModel* node_set)
  {
    auto node = ownedNodes[owned_i];
    apf::MeshEntity* e = node.entity;
    std::string const& NS_name = node_set->stkName;
    std::vector<double>& ns_coords = nodeset_node_coords[NS_name];
    assert(ns_coords.size() >= (nodeSetSizes[NS_name] + 1) * mesh_dim);
    double* node_coords = &(ns_coords.at(nodeSetSizes[NS_name] * mesh_dim));
    nodeSetCoords[NS_name].push_back(node_coords);
    double buf[3];
    apf::getComponents(m->getCoordinateField(), e, node.node, buf);
    for (int j = 0; j < mesh_dim; ++j) node_coords[j] = buf[j];
    nodeSets[NS_name].push_back(std::vector<int>());
    std::vector<int>& dofLids = nodeSets[NS_name].back();
    dofLids.resize(neq);
    for (std::size_t eq=0; eq < neq; eq++)
      dofLids[eq] = getDOF(owned_i, eq);
    ++(nodeSetSizes[NS_name]);
  };
  forEachNodeSetNode(fill_fn);
}

void APFDiscretization::computeSideSets()
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

    SideStruct sstruct;

    sstruct.elem_GID = apf::getNumber(elementNumbering, apf::Node(elem, 0));
    int workset = elemGIDws[sstruct.elem_GID].ws; // workset ID that this element lives in
    sstruct.elem_LID = elemGIDws[sstruct.elem_GID].LID; // local element id in this workset
    sstruct.elem_ebIndex = meshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[workset]]; // element block that workset lives in
    sstruct.side_local_id = apf::getLocalSideId(m, elem, side);

    SideSetList& ssList = sideSets[workset]; // Get a ref to the side set map for this ws

    // Get an iterator to the correct sideset (if it exists)
    SideSetList::iterator sit = ssList.find(ss_name);

    if (sit != ssList.end()) // The sideset has already been created
      sit->second.push_back(sstruct); // Save this side to the vector that belongs to the name ss->first
    else { // Add the key ss_name to the map, and the side vector to that map
      std::vector<SideStruct> tmpSSVec;
      tmpSSVec.push_back(sstruct);
      ssList.insert(SideSetList::value_type(ss_name, tmpSSVec));
    }
  }
  m->end(it);
}

void APFDiscretization::
copyQPScalarToAPF(unsigned nqp,
                  const std::string& stateName,
                  apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        apf::setScalar(f,buck[e],p,ar(e,p));
    }}
  }
}

void APFDiscretization::
copyQPVectorToAPF(unsigned nqp,
                  const std::string& stateName,
                  apf::Field* f)
{
  const unsigned spdim = meshStruct->problemDim;
  apf::Vector3 v(0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < spdim; ++i) {
          v[i] = ar(e,p,i);
        }
        apf::setVector(f,buck[e],p,v);
      }
    }
  }
}

void APFDiscretization::
copyQPTensorToAPF(unsigned nqp,
                  const std::string& stateName,
                  apf::Field* f)
{
  const unsigned spdim = meshStruct->problemDim;
  apf::Matrix3x3 v(0,0,0,0,0,0,0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        for (std::size_t i=0; i < spdim; ++i) {
          for (std::size_t j=0; j < spdim; ++j) {
            v[i][j] = ar(e,p,i,j);
        }}
        apf::setMatrix(f,buck[e],p,v);
      }
    }
  }
}

void APFDiscretization::
copyQPStatesToAPF(apf::FieldShape* fs,
                  bool copyAll)
{
  apf::Mesh2* m = meshStruct->getMesh();
  for (std::size_t i=0; i < meshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(meshStruct->qpscalar_states[i]);
    if (!copyAll && !state.output) {
      continue;
    }
    int nqp = state.dims[1];
    auto f = apf::createField(m,state.name.c_str(),apf::SCALAR,fs);
    copyQPScalarToAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(meshStruct->qpvector_states[i]);
    if (!copyAll && !state.output) {
      continue;
    }
    int nqp = state.dims[1];
    auto f = apf::createField(m,state.name.c_str(),apf::VECTOR,fs);
    copyQPVectorToAPF(nqp, state.name, f);
  }
  for (std::size_t i=0; i < meshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(meshStruct->qptensor_states[i]);
    if (!copyAll && !state.output) {
      continue;
    }
    int nqp = state.dims[1];
    auto f = apf::createField(m,state.name.c_str(),apf::MATRIX,fs);
    copyQPTensorToAPF(nqp, state.name, f);
  }
  if (meshStruct->saveStabilizedStress) {
    saveStabilizedStress();
  }
}

void APFDiscretization::removeQPStatesFromAPF()
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

void APFDiscretization::
copyQPScalarFromAPF(unsigned nqp,
                    const std::string& stateName,
                    apf::Field* f)
{
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p = 0; p < nqp; ++p) {
        ar(e,p) = apf::getScalar(f,buck[e],p);
  }}}
}

void APFDiscretization::
copyQPVectorFromAPF(unsigned nqp,
                    const std::string& stateName,
                    apf::Field* f)
{
  const unsigned spdim = meshStruct->problemDim;
  apf::Vector3 v(0,0,0);
  for (std::size_t b=0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getVector(f,buck[e],p,v);
        for (std::size_t i=0; i < spdim; ++i) {
          ar(e,p,i) = v[i];
  }}}}
}

void APFDiscretization::
copyQPTensorFromAPF(unsigned nqp,
                    const std::string& stateName,
                    apf::Field* f)
{
  const unsigned spdim = meshStruct->problemDim;
  apf::Matrix3x3 v(0,0,0,0,0,0,0,0,0);
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    std::vector<apf::MeshEntity*>& buck = buckets[b];
    MDArray& ar = stateArrays.elemStateArrays[b][stateName];
    for (std::size_t e=0; e < buck.size(); ++e) {
      for (std::size_t p=0; p < nqp; ++p) {
        apf::getMatrix(f,buck[e],p,v);
        for (std::size_t i=0; i < spdim; ++i) {
          for (std::size_t j=0; j < spdim; ++j) {
            ar(e,p,i,j) = v[i][j];
  }}}}}
}

void APFDiscretization::copyQPStatesFromAPF()
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

void APFDiscretization::
copyNodalDataToAPF (const bool copy_all)
{
  if (meshStruct->nodal_data_base.is_null()) { return; }

  auto& node_states = *meshStruct->nodal_data_base->getNodeContainer();

  for (auto& nfs : node_states) {
    Teuchos::RCP<PUMINodeDataBase<RealType> >
      nd = Teuchos::rcp_dynamic_cast<PUMINodeDataBase<RealType>>(
        nfs.second);
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
    if (!PCU_Comm_Self()) std::cerr << "setting nodal field " << nd->name;
    PCU_Barrier();
    setField(nd->name.c_str(), &nd->buffer[0], false, 0, false);
  }
}

void APFDiscretization::removeNodalDataFromAPF ()
{
  if (meshStruct->nodal_data_base.is_null()) { return; }

  apf::Mesh2* m = meshStruct->getMesh();
  auto& node_states = *meshStruct->nodal_data_base->getNodeContainer();

  for (auto& nfs : node_states) {
    Teuchos::RCP<PUMINodeDataBase<RealType> >
      nd = Teuchos::rcp_dynamic_cast<PUMINodeDataBase<RealType>>(
        nfs.second);
    apf::destroyField(m->findField(nd->name.c_str()));
  }
}

void APFDiscretization::
initTimeFromParamLib(Teuchos::RCP<ParamLib> paramLib)
{
  for (std::size_t b = 0; b < buckets.size(); ++b) {
    if (stateArrays.elemStateArrays[b].count("Time")) {
      TEUCHOS_TEST_FOR_EXCEPTION(
        !paramLib->isParameter("Time"), std::logic_error,
        "APF: Time is a state but not a parameter, cannot reinitialize it\n");
      MDArray& time = stateArrays.elemStateArrays[b]["Time"];
      time(0) = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    }
    if (stateArrays.elemStateArrays[b].count("Time_old")) {
      MDArray& oldTime = stateArrays.elemStateArrays[b]["Time_old"];
      oldTime(0) = paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
    }
  }
}

void APFDiscretization::initMesh()
{
  TEUCHOS_FUNC_TIME_MONITOR("APFDiscretization::initMesh");
  computeOwnedNodesAndUnknowns();
  computeOverlapNodesAndUnknowns();
  setupMLCoords();
  computeGraphs();
  getCoordinates(); //fill the coordinates array
  computeWorksetInfo();
  computeNodeSets();
  computeSideSets();

  // load the LandIce Data and tell the state manager to not initialize
  // these fields
  if (meshStruct->shouldLoadLandIceData) {
    setLandIceData();
  }

  // Tell the nodal data base that the graph changed. We don't create the graph
  // (as STKDiscretization does), but others might (such as
  // ProjectIPtoNodalField), so invalidate it.
  if (Teuchos::nonnull(meshStruct->nodal_data_base)) {
    meshStruct->nodal_data_base->updateNodalGraph(Teuchos::null);
  }

  apf::destroyGlobalNumbering(globalNumbering);
  globalNumbering = 0;
  apf::destroyGlobalNumbering(elementNumbering);
  elementNumbering = 0;
}

void APFDiscretization::
updateMesh(bool shouldTransferIPData,
           Teuchos::RCP<ParamLib> paramLib)
{
  TEUCHOS_FUNC_TIME_MONITOR("APFDiscretization::updateMesh");
  initMesh();

  // transfer of internal variables
  if (shouldTransferIPData) {
    copyQPStatesFromAPF();
  }

  // Use the parameter library to re-initialize Time state arrays
  if (Teuchos::nonnull(paramLib)) {
    initTimeFromParamLib(paramLib);
  }
}

void APFDiscretization::attachQPData() {
  int order = meshStruct->cubatureDegree;
  int dim = meshStruct->getMesh()->getDimension();
  apf::FieldShape* fs = apf::getVoronoiShape(dim,order);
  assert(fs);
  copyQPStatesToAPF(fs);
}

void APFDiscretization::detachQPData() {
  removeQPStatesFromAPF();
}

static apf::Field* interpolate(apf::Field* nf,
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

static apf::Field* try_interpolate(apf::Mesh* m,
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

static void temperaturesToQP(apf::Mesh* m,
                             int cubatureDegree)
{
  int o = cubatureDegree;
  if (!try_interpolate(m, "temp", o, "Temperature")) {
    try_interpolate(m, APFMeshStruct::solution_name[0], o, "Temperature");
  }
  if (!try_interpolate(m, "temp_old", o, "Temperature_old")) {
    if (!try_interpolate(m, "temp", o, "Temperature_old")) {
      try_interpolate(m, APFMeshStruct::solution_name[0], o, "Temperature_old");
  }}
}

/* LCM's ThermoMechanicalCoefficients evaluator
   relies on Temperature and Temperature_old
   to be initialized in the stateArrays as well
   as the solution vector.
   this hack will interpolate values from the
   solution vector "temp" to populate the
   stateArrays */

void APFDiscretization::initTemperatureHack()
{
  if (!meshStruct->useTemperatureHack) {
    return;
  }
  apf::Mesh* m = meshStruct->getMesh();
  temperaturesToQP(m, meshStruct->cubatureDegree);
  copyQPStatesFromAPF();
  apf::destroyField(m->findField("Temperature"));
  apf::destroyField(m->findField("Temperature_old"));
}

} // namespace Albany
