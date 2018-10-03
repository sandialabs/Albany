//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTMESHSTRUCT_HPP
#define ALBANY_ABSTRACTMESHSTRUCT_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_CommTypes.hpp"

#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include "Shards_CellTopology.hpp"
#include "Albany_Layouts.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Adapt_NodalDataBase.hpp"


namespace Albany {

enum class LayeredMeshOrdering
{
  LAYER  = 0,
  COLUMN = 1
};

template <typename T>
struct LayeredMeshNumbering {
  T stride;

  LayeredMeshOrdering ordering;
  Teuchos::ArrayRCP<double> layers_ratio;
  T numLevels, numLayers;

  LayeredMeshNumbering(const T _stride, const LayeredMeshOrdering _ordering, const Teuchos::ArrayRCP<double>& _layers_ratio){
    stride = _stride;
    ordering = _ordering;
    layers_ratio= _layers_ratio;
    numLayers = layers_ratio.size();
    numLevels = numLayers+1;
  }

  T getId(const T column_id, const T level_index) const {
      return  (ordering == LayeredMeshOrdering::LAYER) ?
          column_id + level_index*stride :
          column_id * stride + level_index;
  }

  void getIndices(const T id, T& column_id, T& level_index) const {
    if(ordering == LayeredMeshOrdering::COLUMN)  {
      level_index = id%stride;
      column_id = id/stride;
    } else {
      level_index = id/stride;
      column_id = id%stride;
    }
  }
};

struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    enum { DEFAULT_WORKSET_SIZE = 1000 };

    //! Internal mesh specs type needed
    enum msType {
      STK_MS,
#ifdef ALBANY_SCOREC
      PUMI_MS,
#endif
    };

    virtual void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {}) = 0;

    virtual Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() = 0;
    virtual const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() const = 0;

    virtual msType meshSpecsType() = 0;

    Teuchos::RCP<LayeredMeshNumbering<LO> > layered_mesh_numbering;

    Teuchos::RCP<Adapt::NodalDataBase> nodal_data_base;

};

} // Namespace Albany

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
