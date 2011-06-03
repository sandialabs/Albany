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


#ifndef ALBANY_LINE1D_STKMESHSTRUCT_HPP
#define ALBANY_LINE1D_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"


namespace Albany {

  class Line1DSTKMeshStruct : public GenericSTKMeshStruct {

    public:

    Line1DSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& comm);

    ~Line1DSTKMeshStruct() {};

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_,
                  const unsigned int worksetSize);

    private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    bool periodic;

    int nelem;
    Teuchos::RCP<Epetra_Map> elem_map;
  };

}

#endif // ALBANY_LINE1D_STKMESHSTRUCT_HPP
