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
get_dof_mgr (const std::string& part_name,
                    const FE_Type fe_type,
                    const int order,
                    const int dof_dim)
 -> dof_mgr_ptr_t&
{
  // NOTE: we assume order<10, and dof_dim<10, which is virtually never going to change
  int type_order_dim = 100*static_cast<int>(fe_type) + 10*order + dof_dim;

  std::string key = part_name + "_" + std::to_string(type_order_dim);
  return m_key_to_dof_mgr[key];
}

} // namepace Albany
