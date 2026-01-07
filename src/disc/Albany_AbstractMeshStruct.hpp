//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTMESHSTRUCT_HPP
#define ALBANY_ABSTRACTMESHSTRUCT_HPP

#include "Albany_CommTypes.hpp"

#include "Albany_StateInfoStruct.hpp"
#include "Albany_MeshSpecs.hpp"
#include "Albany_LayeredMeshNumbering.hpp"
#include "Albany_AbstractMeshFieldAccessor.hpp"

#include "Shards_CellTopology.hpp"
#include "Albany_Layouts.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    enum { DEFAULT_WORKSET_SIZE = 1000 };

    //! Internal mesh specs type needed
    virtual std::string meshLibName() const = 0;

    virtual void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                               const Teuchos::RCP<StateInfoStruct>& sis,
                               std::map<std::string, Teuchos::RCP<StateInfoStruct> > side_set_sis) = 0;

    virtual void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm) = 0;

    bool isBulkDataSet () const { return m_bulk_data_set; }
    bool isFieldDataSet () const { return m_field_data_set; }

    virtual LO get_num_local_elements () const = 0;
    virtual LO get_num_local_nodes () const = 0;
    virtual GO get_max_node_gid () const = 0;
    virtual GO get_max_elem_gid () const = 0;

    virtual Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor() const = 0;

    Teuchos::RCP<LayeredMeshNumbering<GO> > global_cell_layers_data;
    Teuchos::RCP<LayeredMeshNumbering<LO> > local_cell_layers_data;
    Teuchos::ArrayRCP<double> mesh_layers_ratio;

    Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > meshSpecs;
    std::map<std::string, Teuchos::RCP<AbstractMeshStruct>> sideSetMeshStructs;
  protected:

    bool m_field_data_set = false;
    bool m_bulk_data_set = false;
};

} // Namespace Albany

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
