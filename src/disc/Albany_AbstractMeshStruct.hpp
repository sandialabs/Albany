//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_MESH_STRUCT_HPP
#define ALBANY_ABSTRACT_MESH_STRUCT_HPP

#include "Albany_CommTypes.hpp"

#include "Albany_StateInfoStruct.hpp"
#include "Albany_MeshSpecs.hpp"
#include "Albany_LayeredMeshNumbering.hpp"

#include "Albany_NodalDataBase.hpp"

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

struct AbstractMeshStruct {
public:

  virtual ~AbstractMeshStruct() = default;

  enum { DEFAULT_WORKSET_SIZE = 1000 };

  virtual std::string type () const = 0;

  virtual void setFieldData(
                const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::RCP<StateInfoStruct>& sis,
                const unsigned int worksetSize,
                const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis = {}) = 0;

  virtual void setBulkData(
                const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::RCP<StateInfoStruct>& sis,
                const unsigned int worksetSize,
                const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis = {}) = 0;

  void setFieldAndBulkData(
                const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::RCP<StateInfoStruct>& sis,
                const unsigned int worksetSize,
                const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis = {})
  {
    setFieldData(comm, sis, worksetSize, side_set_sis);
    setBulkData(comm, sis, worksetSize, side_set_sis);
  }

  Teuchos::RCP<MeshSpecs> meshSpecs;

  Teuchos::RCP<LayeredMeshNumbering<GO> > global_cell_layers_data;
  Teuchos::RCP<LayeredMeshNumbering<LO> > local_cell_layers_data;
  Teuchos::ArrayRCP<double> mesh_layers_ratio;

  Teuchos::RCP<NodalDataBase> nodal_data_base;
};

} // Namespace Albany

#endif // ALBANY_ABSTRACT_MESH_STRUCT_HPP
