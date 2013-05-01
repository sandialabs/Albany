//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTSTKFIELDCONT_HPP
#define ALBANY_ABSTRACTSTKFIELDCONT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

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


    typedef stk::mesh::Field<double,stk::mesh::Cartesian,stk::mesh::Cartesian> TensorFieldType ;
    typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType ;
    typedef stk::mesh::Field<double>                      ScalarFieldType ;
    typedef stk::mesh::Field<int>                      IntScalarFieldType ;

    typedef stk::mesh::Cartesian QPTag; // need to invent shards::ArrayDimTag
    typedef stk::mesh::Field<double,QPTag, stk::mesh::Cartesian,stk::mesh::Cartesian> QPTensorFieldType ;
    typedef stk::mesh::Field<double,QPTag, stk::mesh::Cartesian > QPVectorFieldType ;
    typedef stk::mesh::Field<double,QPTag>                      QPScalarFieldType ;

    typedef std::vector<std::string> ScalarValueState;
    typedef std::vector<QPScalarFieldType*> QPScalarState;
    typedef std::vector<QPVectorFieldType*> QPVectorState;
    typedef std::vector<QPTensorFieldType*> QPTensorState;
  
    //! Destructor
    virtual ~AbstractSTKFieldContainer() {};

    VectorFieldType* getCoordinatesField(){ return coordinates_field; }
    IntScalarFieldType* getProcRankField(){ return proc_rank_field; }
    ScalarFieldType* getSurfaceHeightField(){ return surfaceHeight_field; }
    VectorFieldType* getSolutionField(){ return solution_field; }

    ScalarValueState getScalarValueStates(){ return scalarValue_states;}
    QPScalarState getQPScalarStates(){return qpscalar_states;}
    QPVectorState getQPVectorStates(){return qpvector_states;}
    QPTensorState getQPTensorStates(){return qptensor_states;}

    virtual double *getSolutionFieldData(const stk::mesh::Entity&) = 0;
    virtual double *getResidualFieldData(const stk::mesh::Entity&) = 0;

    bool hasResidualField(){ return (residual_field != NULL); }

    double& getTime(){ return time;}

  protected:

    VectorFieldType* coordinates_field;
    IntScalarFieldType* proc_rank_field;
    VectorFieldType* solution_field;
    VectorFieldType* residual_field;
    ScalarFieldType* surfaceHeight_field; // Required for FELIX

    ScalarValueState scalarValue_states;
    QPScalarState qpscalar_states;
    QPVectorState qpvector_states;
    QPTensorState qptensor_states;

    double time;

    int neq;
    int numDim;

  };


}

#endif // ALBANY_ABSTRACTSTKFIELDCONT_HPP
