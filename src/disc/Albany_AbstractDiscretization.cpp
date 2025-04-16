#include "Albany_AbstractDiscretization.hpp"

namespace Albany
{

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
