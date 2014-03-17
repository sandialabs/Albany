//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACTMESHSTRUCT_HPP
#define ALBANY_ABSTRACTMESHSTRUCT_HPP

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include "Shards_CellTopology.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Adapt_NodalDataBlock.hpp"


namespace Albany {

template <typename T>
struct DynamicDataArray {
   typedef Teuchos::ArrayRCP<Teuchos::RCP<T> > type;
};

class CellSpecs {

  public:

    CellSpecs(const CellTopologyData &ctd, const int worksetSize, const int cubdegree, 
                  const int numdim, const int vecdim = -1, const int numface = 0, bool compositeTet = false) :
        cellTopologyData(ctd),
        cellType(shards::CellTopology (&ctd)),
        intrepidBasis(Albany::getIntrepidBasis(ctd, compositeTet)),
        cellCubature(cubFactory.create(cellType, cubdegree)),
        dl(worksetSize, cellType.getNodeCount(),
                          intrepidBasis->getCardinality(), cellCubature->getNumPoints(), numdim, vecdim, numface)
     { }

     unsigned int getNumVertices(){ return cellType.getNodeCount(); }
     unsigned int getNumNodes(){ return intrepidBasis->getCardinality(); }
     unsigned int getNumQPs(){ return cellCubature->getNumPoints(); }

   private:

     static Intrepid::DefaultCubatureFactory<RealType> cubFactory;

     const CellTopologyData &cellTopologyData; // Information about the topology of the elements contained in the workset
     const shards::CellTopology cellType; // the topology of the elements contained in the workset
     const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis; // The basis
     const Teuchos::RCP<Intrepid::Cubature<RealType> > cellCubature; // The cubature of the cells in the workset
     // Make sure this appears after the above, as it depends on the above being initialized prior to
     // dl being initialized
     const Albany::Layouts dl; // the data layout for the elements in the workset

};


struct AbstractMeshStruct {

    virtual ~AbstractMeshStruct() {}

  public:

    //! Internal mesh specs type needed
#ifdef ALBANY_SCOREC
    enum msType { STK_MS, FMDB_VTK_MS, FMDB_EXODUS_MS };
#else
    enum msType { STK_MS };
#endif

    virtual void setFieldAndBulkData(
      const Teuchos::RCP<const Epetra_Comm>& comm,
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const unsigned int neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
      const unsigned int worksetSize) = 0;

    virtual Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() = 0;

    virtual const Albany::DynamicDataArray<Albany::CellSpecs>::type& getMeshDynamicData() const = 0;

    virtual msType meshSpecsType() = 0;

    Teuchos::RCP<Adapt::NodalDataBlock> nodal_data_block;

};
}

#endif // ALBANY_ABSTRACTMESHSTRUCT_HPP
