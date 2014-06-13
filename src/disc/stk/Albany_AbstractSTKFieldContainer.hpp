//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTSTKFIELDCONT_HPP
#define ALBANY_ABSTRACTSTKFIELDCONT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/fem/CoordinateSystems.hpp>

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
class AbstractSTKFieldContainer : public AbstractFieldContainer {

  public:


    // Tensor per Node  - (Node, Dim, Dim)
    typedef stk_classic::mesh::Field<double, stk_classic::mesh::Cartesian, stk_classic::mesh::Cartesian> TensorFieldType ;
    // Vector per Node  - (Node, Dim)
    typedef stk_classic::mesh::Field<double, stk_classic::mesh::Cartesian> VectorFieldType ;
    // One double scalar per Node  - (Node)
    typedef stk_classic::mesh::Field<double>                      ScalarFieldType ;
    // One int scalar per Node  - (Node)
    typedef stk_classic::mesh::Field<int>                         IntScalarFieldType ;

    typedef stk_classic::mesh::Cartesian QPTag; // need to invent shards::ArrayDimTag
    // Tensor per QP   - (Cell, QP, Dim, Dim)
    typedef stk_classic::mesh::Field<double, QPTag, stk_classic::mesh::Cartesian, stk_classic::mesh::Cartesian> QPTensorFieldType ;
    // Vector per QP   - (Cell, QP, Dim)
    typedef stk_classic::mesh::Field<double, QPTag, stk_classic::mesh::Cartesian > QPVectorFieldType ;
    // One scalar per QP   - (Cell, QP)
    typedef stk_classic::mesh::Field<double, QPTag>                      QPScalarFieldType ;

    typedef std::vector<std::string> ScalarValueState;
    typedef std::vector<QPScalarFieldType*> QPScalarState;
    typedef std::vector<QPVectorFieldType*> QPVectorState;
    typedef std::vector<QPTensorFieldType*> QPTensorState;

    typedef std::vector<ScalarFieldType*> ScalarState;
    typedef std::vector<VectorFieldType*> VectorState;
    typedef std::vector<TensorFieldType*> TensorState;

    //! Destructor
    virtual ~AbstractSTKFieldContainer() {};

    VectorFieldType* getCoordinatesField(){ return coordinates_field; }
    IntScalarFieldType* getProcRankField(){ return proc_rank_field; }
    IntScalarFieldType* getRefineField(){ return refine_field; }
#ifdef ALBANY_LCM
    IntScalarFieldType* getFractureState(){ return fracture_state; }
#endif // ALBANY_LCM
    ScalarFieldType* getSurfaceHeightField(){ return surfaceHeight_field; }
    ScalarFieldType* getTemperatureField(){ return temperature_field; }
    ScalarFieldType* getBasalFrictionField(){ return basalFriction_field; }
    ScalarFieldType* getThicknessField(){ return thickness_field; }
    ScalarFieldType* getFlowFactorField(){ return flowFactor_field; }
    VectorFieldType* getSurfaceVelocityField(){ return surfaceVelocity_field; }
    VectorFieldType* getVelocityRMSField(){ return velocityRMS_field; }
    ScalarFieldType* getSphereVolumeField(){ return sphereVolume_field; }

    ScalarValueState getScalarValueStates(){ return scalarValue_states;}
    QPScalarState getQPScalarStates(){return qpscalar_states;}
    QPVectorState getQPVectorStates(){return qpvector_states;}
    QPTensorState getQPTensorStates(){return qptensor_states;}

    virtual bool hasResidualField() = 0;
    virtual bool hasSurfaceHeightField() = 0;
    virtual bool hasTemperatureField() = 0;
    virtual bool hasBasalFrictionField() = 0;
    virtual bool hasThicknessField() = 0;
    virtual bool hasFlowFactorField() = 0;
    virtual bool hasSurfaceVelocityField() = 0;
    virtual bool hasVelocityRMSField() = 0;
    virtual bool hasSphereVolumeField() = 0;

    std::map<std::string, double>& getTime() {
      return time;
    }

    virtual void fillSolnVector(Epetra_Vector& soln, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    virtual void saveSolnVector(const Epetra_Vector& soln, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    virtual void saveResVector(const Epetra_Vector& res, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;

    virtual void transferSolutionToCoords() = 0;

  protected:

    VectorFieldType* coordinates_field;
    IntScalarFieldType* proc_rank_field;
    IntScalarFieldType* refine_field;
#ifdef ALBANY_LCM
    IntScalarFieldType* fracture_state;
#endif // ALBANY_LCM
    ScalarFieldType* surfaceHeight_field; // Required for FELIX
    ScalarFieldType* temperature_field; // Required for FELIX
    ScalarFieldType* basalFriction_field; // Required for FELIX
    ScalarFieldType* thickness_field; // Required for FELIX
    ScalarFieldType* flowFactor_field; // Required for FELIX
    VectorFieldType* surfaceVelocity_field; // Required for FELIX
    VectorFieldType* velocityRMS_field; // Required for FELIX
    ScalarFieldType* sphereVolume_field; // Required for Peridynamics in LCM

    ScalarValueState scalarValue_states;
    QPScalarState qpscalar_states;
    QPVectorState qpvector_states;
    QPTensorState qptensor_states;

    std::map<std::string, double> time;

};

}

#endif // ALBANY_ABSTRACTSTKFIELDCONT_HPP
