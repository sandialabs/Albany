//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_FMDBMESHSTRUCT_HPP
#define ALBANY_FMDBMESHSTRUCT_HPP

#include "Albany_AbstractMeshStruct.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "EpetraExt_MultiComm.h"

#include "FMDB.h"
#ifdef SCOREC_ACIS
#include "AcisModel.h"
#endif
#ifdef SCOREC_PARASOLID
#include "ParasolidModel.h"
#endif

#define NG_EX_ENTITY_TYPE_MAX 15
#define ENT_DIMS 4

namespace Albany {

  class FMDBMeshStruct : public AbstractMeshStruct {

    public:

    FMDBMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& epetra_comm);

    ~FMDBMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();

    msType meshSpecsType(){ return FMDB_MS; }
    pMeshMdl getMesh() { return mesh; }

    bool hasRestartSolution;
    double restartDataTime;
    int neq;
    bool interleavedOrdering;

    private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    const CellTopologyData *getCellTopologyData(const FMDB_EntTopo topo);

    //! Utility function that uses some integer arithmetic to choose a good worksetSize
    int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

    Teuchos::RCP<Teuchos::FancyOStream> out;

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;

    // Info to map element block to physics set
    bool allElementBlocksHaveSamePhysics;
    std::map<std::string, int> ebNameToIndex;

    pGModel model;
    pMeshMdl mesh;
    bool useSerialMesh;

  };

}
#endif
