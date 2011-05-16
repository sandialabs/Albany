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


#ifndef ALBANY_GENERICSTKMESHSTRUCT_HPP
#define ALBANY_GENERICSTKMESHSTRUCT_HPP

//#include <vector>
//#include <string>
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"


namespace Albany {

  class GenericSTKMeshStruct : public AbstractSTKMeshStruct {

    public:
    virtual void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_,
                  const unsigned int worksetSize) { cout << "Error: In GenericSTKMeshStruct setFieldAndBulkData" << endl; throw 3;};

    const Teuchos::RCP<Albany::MeshSpecsStruct>& getMeshSpecs() const;

    protected: 
    GenericSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const int numDim=-1);

    void SetupFieldData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_, const int nstates_,
                  const int worksetSize_);

    void DeclareParts(std::vector<std::string> nsNames);

    ~GenericSTKMeshStruct();

    Teuchos::RCP<Teuchos::ParameterList> getValidGenericSTKParameters(
         std::string listname = "Discretization Param Names") const;

    Teuchos::RCP<Teuchos::ParameterList> params;
    Teuchos::RCP<Albany::MeshSpecsStruct> meshSpecs;
  };

}

#endif
