//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EXTRUDED_DISCRETIZATION_HPP
#define ALBANY_EXTRUDED_DISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ExtrudedMesh.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_NullSpaceUtils.hpp"

#include <utility>
#include <vector>

namespace Albany {

// ====================== STK Discretization ===================== //

class ExtrudedDiscretization : public AbstractDiscretization
{
public:
  //! Constructor
  ExtrudedDiscretization (const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                          const int neq,
                          const Teuchos::RCP<ExtrudedMesh>&        extruded_mesh,
                          const Teuchos::RCP<AbstractDiscretization>& basal_disc,
                          const Teuchos::RCP<const Teuchos_Comm>&     comm,
                          const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
                          const std::map<int, std::vector<std::string>>& sideSetEquations = {});

  //! Destructor
  virtual ~ExtrudedDiscretization() = default;

  //! Get Node set lists (typedef in Albany_DiscretizationUtils.hpp)
  const NodeSetList&      getNodeSets()      const override { return m_nodeSets;      }
  const NodeSetGIDsList&  getNodeSetGIDs()   const override { return m_nodeSetGIDs;   }
  const NodeSetCoordList& getNodeSetCoords() const override { return m_nodeSetCoords; }

  //! Retrieve coordinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>& getCoordinates() const override { return m_nodes_coordinates; }

  //! Print the coordinates for debugging
  void printCoords() const override;

  Teuchos::RCP<AbstractMeshStruct> getMeshStruct() const override { return m_extruded_mesh; }

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const override { return false; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const override { return -1.0; }

  //! After mesh modification, need to update the element connectivity and nodal
  //! coordinates
  void updateMesh() override;

  //! Function that transforms an STK mesh of a unit cube (for LandIce problems)
  void transformMesh();

  //! Get number of spatial dimensions
  int getNumDim() const override { return m_extruded_mesh->meshSpecs[0]->numDim; }

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_Vector> getSolutionField (const bool overlapped = false) const override;

  void getSolutionMV (Thyra_MultiVector& result, bool overlapped) const override;
  void getSolutionDxDp (Thyra_MultiVector& result, bool overlapped) const override;

  void getField (Thyra_Vector& field_vector, const std::string& field_name) const override;
  void setField (const Thyra_Vector& field_vector,
                 const std::string&  field_name,
                 const bool          overlapped = false) override;

  // --- Methods to write solution in the output file --- //

  //! Write the solution to the mesh database.
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const bool overlapped) override;
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const Thyra_Vector& solution_dot,
                                    const bool overlapped) override;
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const Thyra_Vector& solution_dot,
                                    const Thyra_Vector& solution_dotdot,
                                    const bool overlapped) override;
  void writeSolutionMVToMeshDatabase (const Thyra_MultiVector& solution,
                                      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                      const bool overlapped) override;

  //! Write the solution to file. Must call writeSolution first.
  void writeMeshDatabaseToFile (const double        time,
                                const bool          force_write_solution) override;

  Teuchos::RCP<AdaptationData>
  checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                      const Teuchos::RCP<const Thyra_MultiVector>& dxdp) override;

  void adapt (const Teuchos::RCP<AdaptationData>& adaptData) override;

  void setFieldData() override;

protected:

  void getSolutionField(Thyra_Vector& result, bool overlapped) const;

  void setSolutionField (const Thyra_Vector& soln, const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp, const bool overlapped);
  void setSolutionField (const Thyra_Vector& soln,
                         const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                         const Thyra_Vector& soln_dot,
                         const bool          overlapped);
  void setSolutionField (const Thyra_Vector& soln,
                         const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                         const Thyra_Vector& soln_dot,
                         const Thyra_Vector& soln_dotdot,
                         const bool          overlapped);
  void setSolutionFieldMV (const Thyra_MultiVector& solnT,
                           const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                           const bool overlapped);

  void computeCoordinates();
  void createDOFManagers();

  //! Compute jacobian graph
  virtual void computeGraphs();

  //! Process coords for ML
  void setupMLCoords();

  //! Precompute some data divided by workset
  void computeWorksetInfo();

  //! Compute nodesets information
  void computeNodeSets();

  //! Compute sidesets information
  void computeSideSets();

  void buildCellSideNodeNumerationMaps();

  // Create projectors to restrict solution to sidesets
  void buildSideSetProjectors();

  Teuchos::RCP<DOFManager>
  create_dof_mgr (const std::string& part_name,
                  const std::string& field_name,
                  const FE_Type fe_type,
                  const int order,
                  const int dof_dim);

  // ==================== Members =================== //

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm> m_comm;

  Teuchos::RCP<AbstractDiscretization> m_basal_disc;

  //! Equations that are defined only on some side sets of the mesh
  std::map<int, std::vector<std::string>> m_sideSetEquations;

  //! node sets stored as std::map(string ID, int vector of GIDs)
  NodeSetList      m_nodeSets;
  NodeSetGIDsList  m_nodeSetGIDs;
  NodeSetCoordList m_nodeSetCoords;

  // Coordinates spliced together, as [x0,y0,z0,x1,y1,z1,...]
  Teuchos::ArrayRCP<double>   m_nodes_coordinates;

  // Needed to pass coordinates to ML.
  Teuchos::RCP<RigidBodyModes> m_rigid_body_modes;

  // The underlying extruded mesh
  Teuchos::RCP<ExtrudedMesh> m_extruded_mesh;

  // Keep params around, since we may need them after construction
  Teuchos::RCP<Teuchos::ParameterList> m_disc_params;

  // Sideset discretizations
  strmap_t<Teuchos::RCP<Thyra_LinearOp>>    projectors;
  strmap_t<Teuchos::RCP<Thyra_LinearOp>>    ov_projectors;
};

}  // namespace Albany

#endif  // ALBANY_EXTRUDED_DISCRETIZATION_HPP
