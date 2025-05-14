//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_MESH_FIELD_ACCESSOR_HPP
#define ALBANY_ABSTRACT_MESH_FIELD_ACCESSOR_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_DOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

/*
 * Base interface class to access data on mesh
 *
 * The user can call the methods of this class to extract/write
 * data from/to te underlying mesh into/from Thyra (multi) vectors.
 * The (multi) vectors may refer to either owned or overlapped
 * partitions.
 */

class AbstractMeshFieldAccessor
{
public:
  using ValueState               = std::vector<std::string>;
  using MeshScalarState          = std::map<std::string, double>;
  using MeshVectorState          = std::map<std::string, std::vector<double>>;
  using MeshScalarIntegerState   = std::map<std::string, int>;
  using MeshScalarInteger64State = std::map<std::string, GO>;
  using MeshVectorIntegerState   = std::map<std::string, std::vector<int>>;

  using dof_mgr_ptr_t = Teuchos::RCP<const DOFManager>;
  using mv_ptr_t      = Teuchos::RCP<const Thyra_MultiVector>;

  virtual ~AbstractMeshFieldAccessor () = default;

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
  void addStateStructs(const Teuchos::RCP<const StateInfoStruct>& sis) {
    if (Teuchos::nonnull(sis))
      addStateStructs(*sis);
  }

  virtual void addStateStructs(const StateInfoStruct& sis) = 0;

  const StateInfoStruct& getAllSIS()            const { return all_sis;             }
  const StateInfoStruct& getElemSIS()           const { return elem_sis;            }
  const StateInfoStruct& getNodalSIS()          const { return nodal_sis;           }
  const StateInfoStruct& getNodalParameterSIS() const { return nodal_parameter_sis; }

  // Read from mesh methods
  virtual void fillSolnVector (Thyra_Vector&        soln,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void fillVector (Thyra_Vector&        field_vector,
                           const std::string&   field_name,
                           const dof_mgr_ptr_t& field_dof_mgr,
                           const bool           overlapped) = 0;

  virtual void fillSolnMultiVector (Thyra_MultiVector&   soln,
                                    const dof_mgr_ptr_t& sol_dof_mgr,
                                    const bool           overlapped) = 0;
  virtual void fillSolnSensitivity (Thyra_MultiVector&   dxdp,
                                    const dof_mgr_ptr_t& sol_dof_mgr,
                                    const bool           overlapped) = 0;

  // Write to mesh methods
  virtual void saveVector (const Thyra_Vector&  field_vector,
                           const std::string&   field_name,
                           const dof_mgr_ptr_t& field_dof_mgr,
                           const bool           overlapped) = 0;

  virtual void saveSolnVector (const Thyra_Vector& soln,
                               const mv_ptr_t&     soln_dxdp,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;
  virtual void saveSolnVector (const Thyra_Vector&  soln,
                               const mv_ptr_t&      soln_dxdp,
                               const Thyra_Vector&  soln_dot,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void saveSolnVector (const Thyra_Vector&  soln,
                               const mv_ptr_t&      soln_dxdp,
                               const Thyra_Vector&  soln_dot,
                               const Thyra_Vector&  soln_dotdot,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void saveResVector (const Thyra_Vector&  res,
                              const dof_mgr_ptr_t& dof_mgr,
                              const bool          overlapped) = 0;

  virtual void saveSolnMultiVector (const Thyra_MultiVector& soln,
                                    const mv_ptr_t&          soln_dxdp,
                                    const dof_mgr_ptr_t&     node_vs,
                                    const bool          overlapped) = 0;

  ValueState&         getScalarValueStates () { return scalarValue_states; }
  MeshScalarState&    getMeshScalarStates  () { return mesh_scalar_states; }
  MeshVectorState&    getMeshVectorStates  () { return mesh_vector_states; }

  MeshScalarIntegerState&   getMeshScalarIntegerStates   () { return mesh_scalar_integer_states;    }
  MeshScalarInteger64State& getMeshScalarInteger64States () { return mesh_scalar_integer_64_states; }
  MeshVectorIntegerState&   getMeshVectorIntegerStates   () { return mesh_vector_integer_states;    }

protected:
  // This should always include ALL the ones below
  StateInfoStruct all_sis;

  StateInfoStruct elem_sis;               // Fields which will be avail as <Cell[,tags]>
  StateInfoStruct nodal_sis;              // Fields that are associated with Node tags
  StateInfoStruct nodal_parameter_sis;    // Like nodal_sis, but the App will use this to create dist parameters

  ValueState                scalarValue_states;
  MeshScalarState           mesh_scalar_states;
  MeshVectorState           mesh_vector_states;
  MeshScalarIntegerState    mesh_scalar_integer_states;
  MeshScalarInteger64State  mesh_scalar_integer_64_states;
  MeshVectorIntegerState    mesh_vector_integer_states;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_MESH_FIELD_ACCESSOR_HPP
