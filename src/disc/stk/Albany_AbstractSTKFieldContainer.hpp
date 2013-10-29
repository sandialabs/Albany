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

//This include is added in Tpetra branch to get all the necessary 
//Tpetra includes (e.g., Tpetra_Vector.hpp, Tpetra_Map.hpp, etc.) 
#include "Albany_DataTypes.hpp" 

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


    typedef stk::mesh::Field<double, stk::mesh::Cartesian, stk::mesh::Cartesian> TensorFieldType ;
    typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType ;
    typedef stk::mesh::Field<double>                      ScalarFieldType ;
    typedef stk::mesh::Field<int>                         IntScalarFieldType ;

    typedef stk::mesh::Cartesian QPTag; // need to invent shards::ArrayDimTag
    typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian, stk::mesh::Cartesian> QPTensorFieldType ;
    typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian > QPVectorFieldType ;
    typedef stk::mesh::Field<double, QPTag>                      QPScalarFieldType ;

    typedef std::vector<std::string> ScalarValueState;
    typedef std::vector<QPScalarFieldType*> QPScalarState;
    typedef std::vector<QPVectorFieldType*> QPVectorState;
    typedef std::vector<QPTensorFieldType*> QPTensorState;

    //! Destructor
    virtual ~AbstractSTKFieldContainer() {};

    VectorFieldType* getCoordinatesField(){ return coordinates_field; }
    IntScalarFieldType* getProcRankField(){ return proc_rank_field; }
    IntScalarFieldType* getRefineField(){ return refine_field; }
    IntScalarFieldType* getFractureState(){ return fracture_state; }
    ScalarFieldType* getSurfaceHeightField(){ return surfaceHeight_field; }
    ScalarFieldType* getTemperatureField(){ return temperature_field; }
    ScalarFieldType* getBasalFrictionField(){ return basalFriction_field; }
    ScalarFieldType* getThicknessField(){ return thickness_field; }
    ScalarFieldType* getFlowFactorField(){ return flowFactor_field; }

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

    double& getTime() {
      return time;
    }

    virtual void fillSolnVector(Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    virtual void fillSolnVectorT(Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;
    virtual void saveSolnVector(const Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;
    //Tpetra version of above
    virtual void saveSolnVectorT(const Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) = 0;
    virtual void saveResVector(const Epetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) = 0;

    virtual void transferSolutionToCoords() = 0;

  protected:

    VectorFieldType* coordinates_field;
    IntScalarFieldType* proc_rank_field;
    IntScalarFieldType* refine_field;
    IntScalarFieldType* fracture_state;
    ScalarFieldType* surfaceHeight_field; // Required for FELIX
    ScalarFieldType* temperature_field; // Required for FELIX
    ScalarFieldType* basalFriction_field; // Required for FELIX
    ScalarFieldType* thickness_field; // Required for FELIX
    ScalarFieldType* flowFactor_field; // Required for FELIX

    ScalarValueState scalarValue_states;
    QPScalarState qpscalar_states;
    QPVectorState qpvector_states;
    QPTensorState qptensor_states;

    double time;

};

}

#endif // ALBANY_ABSTRACTSTKFIELDCONT_HPP
