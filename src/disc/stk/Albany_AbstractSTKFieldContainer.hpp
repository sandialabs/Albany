//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE set to off.

#ifndef ALBANY_ABSTRACTSTKFIELDCONT_HPP
#define ALBANY_ABSTRACTSTKFIELDCONT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#endif

//This include is added in Tpetra branch to get all the necessary
//Tpetra includes (e.g., Tpetra_Vector.hpp, Tpetra_Map.hpp, etc.)
#include "Albany_DataTypes.hpp"

#include "Albany_NodalDOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
class AbstractSTKFieldContainer : public AbstractFieldContainer {

  public:


    // Tensor per Node/Cell  - (Node, Dim, Dim) or (Cell,Dim,Dim)
    typedef stk::mesh::Field<double, stk::mesh::Cartesian, stk::mesh::Cartesian> TensorFieldType ;
    // Vector per Node/Cell  - (Node, Dim) or (Cell,Dim)
    typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType ;
    // Scalar per Node/Cell  - (Node) or (Cell)
    typedef stk::mesh::Field<double>                      ScalarFieldType ;
    // One int scalar per Node/Cell  - (Node) or (Cell)
    typedef stk::mesh::Field<int>                         IntScalarFieldType ;
    // int vector per Node/Cell  - (Node,Dim/VecDim) or (Cell,Dim/VecDim)
    typedef stk::mesh::Field<int, stk::mesh::Cartesian>   IntVectorFieldType ;

    typedef stk::mesh::Cartesian QPTag; // need to invent shards::ArrayDimTag
    // Tensor3 per QP   - (Cell, QP, Dim, Dim, Dim)
    typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian,
                             stk::mesh::Cartesian,
                             stk::mesh::Cartesian> QPTensor3FieldType ;
    // Tensor per QP   - (Cell, QP, Dim, Dim)
    typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian, stk::mesh::Cartesian> QPTensorFieldType ;
    // Vector per QP   - (Cell, QP, Dim)
    typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian > QPVectorFieldType ;
    // One scalar per QP   - (Cell, QP)
    typedef stk::mesh::Field<double, QPTag>                      QPScalarFieldType ;
    typedef stk::mesh::Field<double, stk::mesh::Cartesian3d>     SphereVolumeFieldType ;

    typedef std::vector<const std::string*> ScalarValueState;
    typedef std::vector<QPScalarFieldType*> QPScalarState;
    typedef std::vector<QPVectorFieldType*> QPVectorState;
    typedef std::vector<QPTensorFieldType*> QPTensorState;
    typedef std::vector<QPTensor3FieldType*> QPTensor3State;

    typedef std::vector<ScalarFieldType*> ScalarState;
    typedef std::vector<VectorFieldType*> VectorState;
    typedef std::vector<TensorFieldType*> TensorState;

    typedef std::map<std::string,double>                MeshScalarState;
    typedef std::map<std::string,std::vector<double> >  MeshVectorState;

    //! Destructor
    virtual ~AbstractSTKFieldContainer() {};

    virtual void addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis) = 0;

    const VectorFieldType* const getCoordinatesField() const { return coordinates_field; }
    VectorFieldType* getCoordinatesField(){ return coordinates_field; }
    IntScalarFieldType* getProcRankField(){ return proc_rank_field; }
    IntScalarFieldType* getRefineField(){ return refine_field; }
#if defined(ALBANY_LCM)
    IntScalarFieldType* getFractureState(stk::topology::rank_t rank){ return fracture_state[rank]; }
#endif // ALBANY_LCM
    SphereVolumeFieldType* getSphereVolumeField(){ return sphereVolume_field; }

    ScalarValueState& getScalarValueStates(){ return scalarValue_states;}
    MeshScalarState& getMeshScalarStates(){return mesh_scalar_states;}
    MeshVectorState& getMeshVectorStates(){return mesh_vector_states;}
    ScalarState& getCellScalarStates(){return cell_scalar_states;}
    VectorState& getCellVectorStates(){return cell_vector_states;}
    TensorState& getCellTensorStates(){return cell_tensor_states;}
    QPScalarState& getQPScalarStates(){return qpscalar_states;}
    QPVectorState& getQPVectorStates(){return qpvector_states;}
    QPTensorState& getQPTensorStates(){return qptensor_states;}
    QPTensor3State& getQPTensor3States(){return qptensor3_states;}
    const StateInfoStruct& getNodalSIS() const {return nodal_sis;}
    const StateInfoStruct& getNodalParameterSIS() const {return nodal_parameter_sis;}

    virtual bool hasResidualField() = 0;
    virtual bool hasSphereVolumeField() = 0;

    std::map<std::string, double>& getTime() {
      return time;
    }

#if defined(ALBANY_EPETRA)
    virtual void fillSolnVector(Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    virtual void fillVector(Epetra_Vector& field_vector, const std::string&  field_name, stk::mesh::Selector& field_selection,
                        const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager) = 0;
    virtual void saveVector(const Epetra_Vector& field_vector, const std::string&  field_name, stk::mesh::Selector& field_selection,
                            const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager) = 0;
    virtual void saveSolnVector(const Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    virtual void saveResVector(const Epetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
#endif
    //Tpetra version of above
    virtual void fillSolnVectorT(Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;
    virtual void fillSolnMultiVector(Tpetra_MultiVector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;
    virtual void saveSolnVectorT(const Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;
    virtual void saveResVectorT(const Tpetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_map) = 0;
    virtual void saveSolnMultiVector(const Tpetra_MultiVector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;

    virtual void transferSolutionToCoords() = 0;

  protected:

    VectorFieldType* coordinates_field;
    IntScalarFieldType* proc_rank_field;
    IntScalarFieldType* refine_field;
#if defined(ALBANY_LCM)
    IntScalarFieldType* fracture_state[stk::topology::ELEMENT_RANK];
#endif // ALBANY_LCM

    SphereVolumeFieldType* sphereVolume_field; // Required for Peridynamics in LCM

    ScalarValueState scalarValue_states;
    MeshScalarState   mesh_scalar_states;
    MeshVectorState   mesh_vector_states;
    ScalarState       cell_scalar_states;
    VectorState       cell_vector_states;
    TensorState       cell_tensor_states;
    QPScalarState     qpscalar_states;
    QPVectorState     qpvector_states;
    QPTensorState     qptensor_states;
    QPTensor3State    qptensor3_states;

    StateInfoStruct nodal_sis;
    StateInfoStruct nodal_parameter_sis;

    std::map<std::string, double> time;

};

}

#endif // ALBANY_ABSTRACTSTKFIELDCONT_HPP
