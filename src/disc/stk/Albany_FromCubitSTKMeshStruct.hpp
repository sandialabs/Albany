//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_FROMCUBIT_STKMESHSTRUCT_HPP
#define ALBANY_FROMCUBIT_STKMESHSTRUCT_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "CUTR_CubitMeshMover.hpp"

#ifdef ALBANY_CUTR

namespace Albany {

  class FromCubitSTKMeshStruct : public AbstractSTKMeshStruct {

    public:

    FromCubitSTKMeshStruct(
                  const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis);


    ~FromCubitSTKMeshStruct();
 
    private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    bool periodic;
  };

}
#endif

#endif
