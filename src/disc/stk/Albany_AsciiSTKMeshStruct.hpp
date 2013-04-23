//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef ALBANY_SEACAS

#ifndef ALBANY_ASCII_STKMESHSTRUCT_HPP
#define ALBANY_ASCII_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
#include <stk_io/MeshReadWriteUtils.hpp>
#include <stk_io/IossBridge.hpp>

#include <Ionit_Initializer.h>

namespace Albany {

  class AsciiSTKMeshStruct : public GenericSTKMeshStruct {

    public:

    AsciiSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Epetra_Comm>& epetra_comm);

    ~AsciiSTKMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    private:
    Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    int NumNodes; //number of nodes
    int NumEles; //number of elements
    double (*xyz)[3]; //hard-coded for 3D for now 
    int (*eles)[8]; //hard-coded for 3D hexes for now 
    Teuchos::RCP<Epetra_Map> elem_map; //element map 
  };

}
#endif
#endif
