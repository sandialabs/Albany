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
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize) = 0;

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();

    protected: 
    GenericSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const int numDim=-1);

    void SetupFieldData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize_);

    void DeclareParts(std::vector<std::string> ebNames, std::vector<std::string> nsNames);

    //! Utility function that uses some integer arithmetic to choose a good worksetSize
    int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

    ~GenericSTKMeshStruct();

    Teuchos::RCP<Teuchos::ParameterList> getValidGenericSTKParameters(
         std::string listname = "Discretization Param Names") const;

    Teuchos::RCP<Teuchos::ParameterList> params;
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;
  };

}

#endif
