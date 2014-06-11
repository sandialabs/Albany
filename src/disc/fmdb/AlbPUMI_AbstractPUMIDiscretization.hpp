//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBPUMI_ABSTRACTPUMIDISCRETIZATION_HPP
#define ALBPUMI_ABSTRACTPUMIDISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"

namespace AlbPUMI {

  class AbstractPUMIDiscretization : public Albany::AbstractDiscretization {
  public:

    //! Destructor
    virtual ~AbstractPUMIDiscretization(){}

    // Retrieve mesh struct
    virtual Teuchos::RCP<AlbPUMI::FMDBMeshStruct> getFMDBMeshStruct() = 0;

    virtual void attachQPData() = 0; 
    virtual void detachQPData() = 0;

    // After mesh modification, need to update the element connectivity and nodal coordinates
    virtual void updateMesh(bool shouldTransferIPData) = 0;

    virtual void debugMeshWriteNative(const Epetra_Vector& sol, const char* filename) = 0;
    virtual void debugMeshWrite(const Epetra_Vector& sol, const char* filename) = 0;

    virtual Teuchos::RCP<const Epetra_Comm> getComm() const = 0;

    virtual void reNameExodusOutput(const std::string& str) = 0;

  };

}

#endif // ALBANY_ABSTRACTPUMIDISCRETIZATION_HPP
