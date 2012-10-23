//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
#define ALBANY_ABSTRACTSTKMESHSTRUCT_HPP

#include <vector>
#include <fstream>

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Albany_StateInfoStruct.hpp"

// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/fem/FEMMetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/fem/CoordinateSystems.hpp>


namespace Albany {
  //! Small container to hold periodicBC info for use in setting coordinates
  struct PeriodicBCStruct {
    PeriodicBCStruct() 
       {periodic[0]=false; periodic[1]=false; periodic[2]=false; 
        scale[0]=1.0; scale[1]=1.0; scale[2]=1.0; };
    bool periodic[3];
    double scale[3];
  };

  struct AbstractSTKMeshStruct {

    //AbstractSTKMeshStruct();
  virtual ~AbstractSTKMeshStruct(){}

  public:

    virtual void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, 
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize) {};


    typedef stk::mesh::Field<double,stk::mesh::Cartesian,stk::mesh::Cartesian> TensorFieldType ;
    typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType ;
    typedef stk::mesh::Field<double>                      ScalarFieldType ;
    typedef stk::mesh::Field<int>                      IntScalarFieldType ;

    typedef stk::mesh::Cartesian QPTag; // need to invent shards::ArrayDimTag
    typedef stk::mesh::Field<double,QPTag, stk::mesh::Cartesian,stk::mesh::Cartesian> QPTensorFieldType ;
    typedef stk::mesh::Field<double,QPTag, stk::mesh::Cartesian > QPVectorFieldType ;
    typedef stk::mesh::Field<double,QPTag>                      QPScalarFieldType ;

    stk::mesh::fem::FEMMetaData* metaData;
    stk::mesh::BulkData* bulkData;
    std::map<int, stk::mesh::Part*> partVec;    //Element blocks
    std::map<std::string, stk::mesh::Part*> nsPartVec;  //Node Sets
    std::map<std::string, stk::mesh::Part*> ssPartVec;  //Side Sets
    VectorFieldType* coordinates_field;
    IntScalarFieldType* proc_rank_field;
    VectorFieldType* solution_field;
    VectorFieldType* residual_field;
    double time;

    std::vector<std::string> scalarValue_states;
    std::vector<QPScalarFieldType*> qpscalar_states;
    std::vector<QPVectorFieldType*> qpvector_states;
    std::vector<QPTensorFieldType*> qptensor_states;

    int numDim;
    int neq;
    bool interleavedOrdering;

    bool exoOutput;
    std::string exoOutFile;
    int exoOutputInterval;
    bool hasRestartSolution;
    double restartDataTime;

    //Flag for transforming STK mesh; currently only needed for FELIX problems 
    std::string transformType;
    //alpha and L are parameters read in from ParameterList for FELIX problems 
    double felixAlpha; 
    int felixL; 

    // Temporary flag to switch between 2D elements being Rank Elements or Faces
    bool useElementAsTopRank;

    // Info to map element block to physics set
    bool allElementBlocksHaveSamePhysics;
    std::map<std::string, int> ebNameToIndex;

    // Info for periodic BCs -- only for hand-coded STK meshes
    struct PeriodicBCStruct PBCStruct;
  };
}

#endif // ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
