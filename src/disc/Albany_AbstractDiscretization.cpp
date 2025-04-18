#include "Albany_AbstractDiscretization.hpp"

namespace Albany
{

void AbstractDiscretization::
updateMesh (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  auto mesh = getMeshStruct();
  if (not mesh->isBulkDataSet()) {
    mesh->setBulkData(comm);
  }

  for (auto& [ss_name, ss_mesh] : mesh->sideSetMeshStructs) {
    // For extruded meshes, the bulk data of the basal mesh
    // should be set from inside the extruded mesh call,
    // during the 'setBulkData' call above
    if (not ss_mesh->isBulkDataSet()) {
      ss_mesh->setBulkData(comm);
    }
    (void)ss_name;
  }

  this->updateMeshImpl(comm);

  // Update sideset discretizations (if any) and build projectors
  for (auto& [ss_name,ss_disc] : sideSetDiscretizations) {
    ss_disc->updateMesh(comm);

    buildSideSetProjectors(ss_name);
  }
}

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const Thyra_Vector& soln_dot,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const Thyra_Vector& soln_dot,
               const Thyra_Vector& soln_dotdot,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolutionMV (const Thyra_MultiVector& soln,
                 const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
                 const double             time,
                 const bool               overlapped,
                 const bool               force_write_solution)
{
  writeSolutionMVToMeshDatabase(soln, soln_dxdp, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

} // namepace Albany
