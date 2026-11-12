//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP
#define ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"

#include "Teuchos_ParameterList.hpp"

// Forward declaration is enough
namespace stk {
namespace mesh {
class BulkData;
class MetaData;
} // namespace stk
} // namespace mesh

namespace Albany {

class GenericSTKFieldContainer : public AbstractSTKFieldContainer
{
public:
  GenericSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                           const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                           const Teuchos::RCP<stk::mesh::BulkData>& bulkData_);

  virtual ~GenericSTKFieldContainer() = default;

  // Add StateStructs to the list of stored ones
  void addStateStruct(const Teuchos::RCP<StateStruct>& st);

  void createStateArrays (const WorksetArray<int>& worksets_sizes);
  void transferNodeStatesToElemStates ();
  void transferElemStateToNodeState (const std::string& name) override;

  Teuchos::RCP<Teuchos::ParameterList> getParams() const {return params; }

  Teuchos::RCP<stk::mesh::MetaData> getMetaData() {return metaData;}
  Teuchos::RCP<stk::mesh::BulkData> getBulkData() {return bulkData;}

protected:

  void setGeometryFieldsMetadata ();

  template<typename T>
  // If ncopies>1, the field has ncmp*ncopies scalars per entity, with the ncmp index being
  // the fastest one (same as layered vector states, which use ncmp=numLayers, ncopies=vecDim)
  stk::mesh::Field<T>* add_field_to_mesh(const std::string& name,
                                         const bool nodal,
                                         const bool transient,
                                         const int ncmp,
                                         const int ncopies = 1);

  Teuchos::RCP<stk::mesh::MetaData> metaData;
  Teuchos::RCP<stk::mesh::BulkData> bulkData;
  Teuchos::RCP<Teuchos::ParameterList> params;

  bool solutionFieldContainer = false;
};

} // namespace Albany

#endif // ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP
