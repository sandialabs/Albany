//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
#define ALBANY_ABSTRACTSTKMESHSTRUCT_HPP

#include <vector>
#include <fstream>

#include "Albany_AbstractMeshStruct.hpp"

#include "Albany_AbstractSTKFieldContainer.hpp"

// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

#include "Teuchos_ScalarTraits.hpp"

namespace Albany {
  //! Small container to hold periodicBC info for use in setting coordinates
  struct PeriodicBCStruct {
    PeriodicBCStruct()
       {periodic[0]=false; periodic[1]=false; periodic[2]=false;
        scale[0]=1.0; scale[1]=1.0; scale[2]=1.0; };
    bool periodic[3];
    double scale[3];
  };

  struct AbstractSTKMeshStruct : public AbstractMeshStruct {

    virtual ~AbstractSTKMeshStruct() {}

  public:

    msType meshSpecsType(){ return STK_MS; }

    Teuchos::RCP<stk::mesh::MetaData> metaData;
    Teuchos::RCP<stk::mesh::BulkData> bulkData;

    std::map<int, stk::mesh::Part*> partVec;    //Element blocks
    std::map<std::string, stk::mesh::Part*> nsPartVec;  //Node Sets
    std::map<std::string, stk::mesh::Part*> ssPartVec;  //Side Sets

    Teuchos::RCP<Albany::AbstractSTKFieldContainer> getFieldContainer(){return fieldContainer; }
    const AbstractSTKFieldContainer::VectorFieldType* const getCoordinatesField() const { return fieldContainer->getCoordinatesField(); }
    AbstractSTKFieldContainer::VectorFieldType* getCoordinatesField(){ return fieldContainer->getCoordinatesField(); }

    int numDim;
    int neq;
    bool interleavedOrdering;

    bool exoOutput;
    std::string exoOutFile;
    int exoOutputInterval;
    std::string cdfOutFile;
    bool cdfOutput;
    unsigned nLat;
    unsigned nLon;
    int cdfOutputInterval;

    bool transferSolutionToCoords;

    // Solution history
    virtual int getSolutionFieldHistoryDepth() const { return 0; } // No history by default
    virtual double getSolutionFieldHistoryStamp(int step) const { return Teuchos::ScalarTraits<double>::nan(); } // Dummy value
    virtual void loadSolutionFieldHistory(int step) { /* Does nothing by default */ }

    //! Flag if solution has a restart values -- used in Init Cond
    virtual bool hasRestartSolution() const = 0;

    //! If restarting, convenience function to return restart data time
    virtual double restartDataTime() const = 0;

    virtual bool useCompositeTet() = 0;

    //Flag for transforming STK mesh; currently only needed for FELIX/Aeras problems
    std::string transformType;
    //alpha and L are parameters read in from ParameterList for FELIX problems
    double felixAlpha;
    double felixL;

    //Points per edge in creating enriched spectral mesh in Aeras::SpectralDiscretization (for Aeras only).
    int points_per_edge;

    bool contigIDs; //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor

    //boolean flag for writing coordinates to matrix market file (e.g., for ML analysis)
    bool writeCoordsToMMFile;

    // Info to map element block to physics set
    bool allElementBlocksHaveSamePhysics;
    std::map<std::string, int> ebNameToIndex;

    // Info for periodic BCs -- only for hand-coded STK meshes
    struct PeriodicBCStruct PBCStruct;

    std::map<std::string,Teuchos::RCP<Albany::AbstractSTKMeshStruct> >  sideSetMeshStructs;

  protected:

    Teuchos::RCP<Albany::AbstractSTKFieldContainer> fieldContainer;

  };
}

#endif // ALBANY_ABSTRACTSTKMESHSTRUCT_HPP
