//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//



#ifndef ALBANY_ABSTRACTPUMIDISCRETIZATION_HPP
#define ALBANY_ABSTRACTPUMIDISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_PUMIMeshStruct.hpp"

namespace Albany {

  class AbstractPUMIDiscretization : public Albany::AbstractDiscretization {
  public:

    //! Destructor
    virtual ~AbstractPUMIDiscretization(){}

    //! Retrieve mesh struct
    virtual Teuchos::RCP<Albany::PUMIMeshStruct> getPUMIMeshStruct() = 0;

    virtual apf::GlobalNumbering* getAPFGlobalNumbering() = 0;

    virtual void attachQPData() = 0;
    virtual void detachQPData() = 0;

    //! After mesh modification, need to update the element connectivity and
    //! nodal coordinates
    virtual void updateMesh(bool shouldTransferIPData) = 0;

    //! There can be situations where we want to create a new apf::Mesh2 from
    //! scratch. Clean up everything that depends on the current mesh first,
    //! thereby releasing the mesh.
    virtual void releaseMesh() = 0;

#if defined(ALBANY_EPETRA)
    virtual void debugMeshWriteNative(const Epetra_Vector& sol, const char* filename) = 0;
    virtual void debugMeshWrite(const Epetra_Vector& sol, const char* filename) = 0;
#endif

    virtual Teuchos::RCP<const Teuchos_Comm> getComm() const = 0;

    virtual void reNameExodusOutput(const std::string& str) = 0;

    //! Create a new field having a name and a value_type of apf::SCALAR,
    //! apf::VECTOR, or apf::MATRIX.
    virtual void createField(const char* name, int value_type) = 0;
    //! Copy field data to APF. nentries is the number of values at each field
    //! point.
    virtual void setField(const char* name, const ST* data, bool overlapped,
                          int offset = 0, int nentries = -1) = 0;
    //! Copy field data from APF.
    virtual void getField(const char* name, ST* dataT, bool overlapped,
                          int offset = 0, int nentries = -1) const = 0;

    //amb-dbg
    virtual void writeMeshDebug (const std::string& filename) {}
  };

}

#endif // ALBANY_ABSTRACTPUMIDISCRETIZATION_HPP
