//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DUMMY_MESH_HPP
#define ALBANY_DUMMY_MESH_HPP

#include "Albany_ExtrudedMesh.hpp"

namespace Albany {

struct DummyExtrudedMesh : public ExtrudedMesh {
public:
  DummyExtrudedMesh (const Teuchos::RCP<const AbstractMeshStruct>& basal_mesh,
                     const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<const Teuchos_Comm>& comm)
   : ExtrudedMesh(basal_mesh,params,comm)
  {
    // Nothing to do here
  }

  //! Internal mesh specs type needed
  std::string meshLibName() const override { return "dummy"; }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
                             const Teuchos::RCP<StateInfoStruct>& /* sis */) override {}

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& /* comm */) override;
};


struct DummyMesh2d : public AbstractMeshStruct {
public:
  DummyMesh2d (const int ne)
   : m_ne (ne)
  {
    // Nothing to do here
  }

  LO get_num_local_elements () const override { return m_ne*m_ne; }

  //! Internal mesh specs type needed
  std::string meshLibName() const override { return "dummy"; }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
                             const Teuchos::RCP<StateInfoStruct>& /* sis */) override {}

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& /* comm */) override;

protected:

  int m_ne;
};

} // Namespace Albany

#endif // ALBANY_DUMMY_MESH_HPP
