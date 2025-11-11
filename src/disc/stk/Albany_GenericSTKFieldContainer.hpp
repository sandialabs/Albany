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
                           const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
                           const int num_params_);

  GenericSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                           const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                           const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
                           const int neq_,
                           const int num_params_);

  virtual ~GenericSTKFieldContainer() = default;

  // Add StateStructs to the list of stored ones
  void addStateStruct(const Teuchos::RCP<StateStruct>& st);

  void createStateArrays (const WorksetArray<int>& worksets_sizes);
  void transferNodeStatesToElemStates ();
  void transferElemStateToNodeState (const std::string& name) override;

  Teuchos::RCP<Teuchos::ParameterList> getParams() const {return params; }

  Teuchos::RCP<stk::mesh::MetaData> getMetaData() {return metaData;}
  Teuchos::RCP<stk::mesh::BulkData> getBulkData() {return bulkData;}

  int getNumParams() const {return num_params; }

protected:

  Teuchos::RCP<stk::mesh::MetaData> metaData;
  Teuchos::RCP<stk::mesh::BulkData> bulkData;
  Teuchos::RCP<Teuchos::ParameterList> params;


  int neq;
  int num_params = 0;

  bool save_solution_field = false;
};

} // namespace Albany

#endif // ALBANY_GENERIC_STK_FIELD_CONTAINER_HPP
