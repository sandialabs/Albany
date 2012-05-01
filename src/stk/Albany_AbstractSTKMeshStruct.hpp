/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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

    // Temporary flag to switch between 2D elements being Rank Elements or Faces
    bool useElementAsTopRank;

    // Info to map element block to physics set
    bool allElementBlocksHaveSamePhysics;
    std::map<std::string, int> ebNameToIndex;
  };
}

#endif // ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
