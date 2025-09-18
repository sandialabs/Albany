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

auto AbstractDiscretization::
get_hashed_dof_mgr (const std::string& part_name,
                    const FE_Type fe_type,
                    const int order,
                    const int dof_dim)
 -> dof_mgr_ptr_t&
{
  std::size_t hash = 0;
  hash ^= std::hash<std::string>()(part_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(order) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(dof_dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(static_cast<int>(fe_type)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return m_hash_to_dof_mgr[hash];
}

} // namepace Albany
