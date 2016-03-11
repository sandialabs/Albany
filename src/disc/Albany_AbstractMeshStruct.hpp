//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTMESHSTRUCT_HPP
#define ALBANY_ABSTRACTMESHSTRUCT_HPP

#include "Teuchos_ParameterList.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include "Shards_CellTopology.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Adapt_NodalDataBase.hpp"


namespace Albany {

template <typename T>
struct DynamicDataArray {
   typedef Teuchos::ArrayRCP<Teuchos::RCP<T> > type;
};

template <typename T>
struct LayeredMeshNumbering {
  T stride;
  bool ordering; //0 for layer wise, 1 for column wise
  Teuchos::ArrayRCP<double> layers_ratio;
  T numLevels, numLayers;

  LayeredMeshNumbering(const T _stride, const bool  _ordering, const Teuchos::ArrayRCP<double>& _layers_ratio){
    stride = _stride;
    ordering = _ordering;
    layers_ratio= _layers_ratio;
    numLayers = layers_ratio.size();
    numLevels = numLayers+1;
  }

  T getId(const T column_id, const T level_index) const {
      return  (ordering == 0) ?
          column_id + level_index*stride :
          column_id * stride + level_index;
  }

  void getIndices(const T id, T& column_id, T& level_index) const {
    if(ordering)  {
      level_index = id%stride;
      column_id = id/stride;
    } else {
      level_index = id/stride;
      column_id = id%stride;
    }
  }
};

class CellSpecs {

  public:

    CellSpecs(const CellTopologyData &ctd, const int worksetSize, const int cubdegree,
                  const int numdim, const int vecdim = -1, const int numface = 0, bool compositeTet = false) :
        cellTopologyData(ctd),
        cellType(shards::CellTopology (&ctd)),
        intrepidBasis(Albany::getIntrepid2Basis(ctd, compositeTet)),
        cellCubature(cubFactory.create(cellType, cubdegree)),
        dl(worksetSize, cellType.getNodeCount(),
                          intrepidBasis->getCardinality(), cellCubature->getNumPoints(), numdim, vecdim, numface)
     { }

     unsigned int getNumVertices(){ return cellType.getNodeCount(); }
     unsigned int getNumNodes(){ return intrepidBasis->getCardinality(); }
     unsigned int getNumQPs(){ return cellCubature->getNumPoints(); }

   private:

     static Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;

     const CellTopologyData &cellTopologyData; // Information about the topology of the elements contained in the workset
     const shards::CellTopology cellType; // the topology of the elements contained in the workset
     const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis; // The basis
     const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cellCubature; // The cubature of the cells in the workset
     // Make sure this appears after the above, as it depends on the above being initialized prior to
     // dl being initialized
     const Albany::Layouts dl; // the data layout for the elements in the workset

};


struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    //! Internal mesh specs type needed
    enum msType {
      STK_MS,
#ifdef ALBANY_SCOREC
      PUMI_MS,
#endif
#ifdef ALBANY_GOAL
      GOAL_MS,
#endif
#ifdef ALBANY_AMP
      SIM_MS,
#endif
    };

    virtual void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize) = 0;
;

    virtual Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() = 0;

    virtual const Albany::DynamicDataArray<Albany::CellSpecs>::type& getMeshDynamicData() const = 0;

    virtual msType meshSpecsType() = 0;

    Teuchos::RCP<LayeredMeshNumbering<LO> > layered_mesh_numbering;

    Teuchos::RCP<Adapt::NodalDataBase> nodal_data_base;

};
}

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
