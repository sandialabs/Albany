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

#include "Shards_CellTopology.hpp"
#include "Albany_Layouts.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Adapt_NodalDataBase.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    enum { DEFAULT_WORKSET_SIZE = 1000 };

    //! Internal mesh specs type needed
    virtual std::string meshType() const = 0;

    virtual void setFieldData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {}) = 0;

    virtual void setBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {}) = 0;

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {})
                  {
                    setFieldData(commT, sis, worksetSize, side_set_sis);
                    setBulkData(commT, sis, worksetSize, side_set_sis);
                  }

    virtual Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() = 0;
    virtual const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() const = 0;

    Teuchos::RCP<LayeredMeshNumbering<GO> > global_cell_layers_data;
    Teuchos::RCP<LayeredMeshNumbering<LO> > local_cell_layers_data;
    Teuchos::ArrayRCP<double> mesh_layers_ratio;

    Teuchos::RCP<Adapt::NodalDataBase> nodal_data_base;
};

} // Namespace Albany

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
