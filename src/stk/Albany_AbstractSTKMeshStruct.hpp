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

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"

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
  public:

    virtual void doFieldsAndBulkData() {};

    typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType ;
    typedef stk::mesh::Field<double>                      ScalarFieldType ;

    stk::mesh::fem::FEMMetaData* metaData;
    stk::mesh::BulkData* bulkData;
    std::map<int, stk::mesh::Part*> partVec;    //Element blocks
    std::map<std::string, stk::mesh::Part*> nsPartVec;  //Node Sets
    VectorFieldType* coordinates_field;
    VectorFieldType* solution_field;
    VectorFieldType* residual_field;
    VectorFieldType* state_field;
    int numDim;
    unsigned int neq;
    unsigned int nstates;

    bool exoOutput;
    std::string exoOutFile;

    int cubatureDegree;

    // Temporary flag to switch between 2D elements being Rank Elements or Faces
    bool useElementAsTopRank;
  };

}

#endif // ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
