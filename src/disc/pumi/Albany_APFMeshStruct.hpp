//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_APFMESHSTRUCT_HPP
#define ALBANY_APFMESHSTRUCT_HPP

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_PUMIQPData.hpp"
#include "Albany_PUMINodeData.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "EpetraExt_MultiComm.h"
#include <PHAL_Dimension.hpp>

#include <apf.h>
#include <apfMesh2.h>
#if defined(HAVE_STK)
#include <apfSTK.h>
#else
#include <apfAlbany.h>
#endif
#include <gmi.h>

namespace Albany {

class APFMeshStruct : public Albany::AbstractMeshStruct {

  public:

    void init(const Teuchos::RCP<Teuchos::ParameterList>& params,
              const Teuchos::RCP<const Teuchos_Comm>& commT);

    virtual ~APFMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    void splitFields(Teuchos::Array<std::string> fieldLayout);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();

    std::vector<Teuchos::RCP<PUMIQPData<double, 1> > > scalarValue_states;
    std::vector<Teuchos::RCP<PUMIQPData<double, 2> > > qpscalar_states;
    std::vector<Teuchos::RCP<PUMIQPData<double, 3> > > qpvector_states;
    std::vector<Teuchos::RCP<PUMIQPData<double, 4> > > qptensor_states;

    std::vector<std::string> nsNames;
    std::vector<std::string> ssNames;

    apf::Mesh2* getMesh() { return mesh; }
    gmi_model* getMdl() { return model; }
    apf::StkModels& getSets() { return sets; }

    // Solution history
    int solutionFieldHistoryDepth;
    void loadSolutionFieldHistory(int step);

    bool useCompositeTet(){ return compositeTet; }

    const Albany::DynamicDataArray<Albany::CellSpecs>::type& getMeshDynamicData() const
        { return meshDynamicData; }

    //! Process PUMI mesh for element block specific info
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

    static const char* solution_name;
    static const char* residual_name;

protected:

    Teuchos::RCP<Teuchos::ParameterList>
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
