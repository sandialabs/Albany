//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACTMESHSTRUCT_HPP
#define ALBANY_ABSTRACTMESHSTRUCT_HPP

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

namespace Albany {

struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    //! Internal mesh specs type needed
    enum msType { STK_MS, FMDB_VTK_MS, FMDB_EXODUS_MS };

    virtual void setFieldAndBulkData(
      const Teuchos::RCP<const Epetra_Comm>& comm,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const unsigned int neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
      const unsigned int worksetSize) = 0;

    virtual Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() = 0;

    virtual msType meshSpecsType() = 0;

};
}

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
