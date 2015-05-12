//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBPUMI_FMDBMESHSTRUCT_HPP
#define ALBPUMI_FMDBMESHSTRUCT_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "AlbPUMI_QPData.hpp"
#include "AlbPUMI_NodeData.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "EpetraExt_MultiComm.h"
#include <PHAL_Dimension.hpp>

#include <apf.h>
#include <apfMesh2.h>
#include <apfMDS.h>
#include <apfSTK.h>
#include <gmi.h>

namespace AlbPUMI {

  class FMDBMeshStruct : public Albany::AbstractMeshStruct {

  public:

    FMDBMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT);

    ~FMDBMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& side_set_sis = Teuchos::null);

    void splitFields(Teuchos::Array<std::string> fieldLayout);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();

    std::vector<Teuchos::RCP<QPData<double, 1> > > scalarValue_states;
    std::vector<Teuchos::RCP<QPData<double, 2> > > qpscalar_states;
    std::vector<Teuchos::RCP<QPData<double, 3> > > qpvector_states;
    std::vector<Teuchos::RCP<QPData<double, 4> > > qptensor_states;

    std::vector<std::string> nsNames;
    std::vector<std::string> ssNames;

    msType meshSpecsType();
    apf::Mesh2* getMesh() { return mesh; }
    gmi_model* getMdl() { return model; }
    apf::StkModels& getSets() { return sets; }

    // Solution history
    int solutionFieldHistoryDepth;
    void loadSolutionFieldHistory(int step);

    bool useCompositeTet(){ return compositeTet; }

    const Albany::DynamicDataArray<Albany::CellSpecs>::type& getMeshDynamicData() const
        { return meshDynamicData; }

    //! Process FMDB mesh for element block specific info
    void setupMeshBlkInfo();

    bool hasRestartSolution;
    double restartDataTime;
    int neq;
    int numDim;
    int cubatureDegree;
    bool interleavedOrdering;
    bool solutionInitialized;
    bool residualInitialized;

    Teuchos::Array<std::string> solVectorLayout;
    Teuchos::Array<std::string> resVectorLayout;

    double time;

    // Info to map element block to physics set
    bool allElementBlocksHaveSamePhysics;
    std::map<std::string, int> ebNameToIndex;

    int worksetSize;

    std::string outputFileName;
    int outputInterval;
    bool useNullspaceTranslationOnly;

private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    //! Utility function that uses some integer arithmetic to choose a good worksetSize
    int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

    Teuchos::RCP<Teuchos::FancyOStream> out;

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;

    // Information that changes when the mesh adapts
    Albany::DynamicDataArray<Albany::CellSpecs>::type meshDynamicData;

    apf::Mesh2* mesh;
    gmi_model* model;
    apf::StkModels sets;

    bool compositeTet;

  };

}
#endif
