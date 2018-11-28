//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/*! \file PeridigmManager.hpp */

#ifndef PERIDIGMMANAGER_HPP
#define PERIDIGMMANAGER_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"

#include <Peridigm.hpp>
#include <Peridigm_AlbanyDiscretization.hpp>

namespace LCM {

class PeridigmManager {

public:

  struct OutputField {
    std::string albanyName;
    std::string peridigmName;
    std::string initType;
    std::string relation;
    int length;
    bool operator==(const OutputField &rhs){
      if(rhs.albanyName != albanyName ||
         rhs.peridigmName != peridigmName ||
         rhs.initType != initType ||
         rhs.relation != relation)
        return false;
      return true;
    }
  };

  struct PartialStressElement {
    stk::mesh::Entity albanyElement;
    CellTopologyData cellTopologyData;
    std::vector<int> peridigmGlobalIds;
    std::vector<RealType> albanyNodeInitialPositions;
    bool operator==(const PartialStressElement &rhs){
      if(rhs.albanyElement != albanyElement ||
         rhs.peridigmGlobalIds != peridigmGlobalIds)
        return false;
      return true;
    }
  };

  //! Data structure for Optimization-Based Coupling
  struct OBCDataPoint {
    double sphereElementVolume;
    double initialCoords[3];
    double currentCoords[3];
    int peridigmGlobalId;
    stk::mesh::Entity albanyElement;
    CellTopologyData cellTopologyData;
    double naturalCoords[3];
  };

  //! Singleton.
  static const Teuchos::RCP<PeridigmManager>& self();
  //! Instantiate the singleton
  static void initializeSingleton(
    const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! Instantiate the Peridigm object
  void initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  Teuchos::RCP<Albany::AbstractDiscretization> disc,
		  const Teuchos::RCP<const Teuchos_Comm>& comm);

  //! Identify the overlapping solid element for each peridynamic sphere element (applies only to overlapping discretizations).
  void obcOverlappingElementSearch();

  //! Evaluate the functional for optimization-based coupling
  double obcEvaluateFunctional(Epetra_Vector* obcFunctionalDerivWrtDisplacement = NULL);

  //! Load the current time and displacement from Albany into the Peridigm manager.
  void setCurrentTimeAndDisplacement(double time, const Teuchos::RCP<const Tpetra_Vector>& albanySolutionVector);
  void setCurrentTimeAndDisplacement(double time, const Teuchos::RCP<const Thyra_Vector>& albanySolutionVector);

  //! Modify Albany graphs for tangent stiffness matrix to include Peridigm nonzeros.
  void insertPeridigmNonzerosIntoAlbanyGraph()
  {
    stkDisc->insertPeridigmNonzerosIntoGraph();
  }

  //! Copy values from the Peridigm tangent stiffness matrix into the Albany jacobian.
  bool copyPeridigmTangentStiffnessMatrixIntoAlbanyJacobian(Teuchos::RCP<Thyra_LinearOp> jac);

  //! Evaluate the peridynamic internal force
  void evaluateInternalForce();

  //! Evaluate the peridynamic tangent stiffness matrix
  void evaluateTangentStiffnessMatrix();

  //! Query existance of the Peridigm tangent stiffness matrix
  bool hasTangentStiffnessMatrix()
  {
    return peridigm->hasTangentStiffnessMatrix();
  }

  //! Access the Peridigm tangent stiffness matrix
  Teuchos::RCP<const Epetra_FECrsMatrix> getTangentStiffnessMatrix();

  //! Update the state within Peridigm following a successful load step.
  void updateState();

  //! Write the Peridigm submodel to a separate Exodus file.
  void writePeridigmSubModel(RealType currentTime);

  //! Retrieve the force for the given global degree of freedom (evaluateInternalForce() must be called prior to getForce()).
  double getForce(int globalAlbanyNodeId, int dof);

  //! Computes a least squares fit of the displacement field for all particles within the horizon of the given node
  double getDisplacementNeighborhoodFit(int globalAlbanyNodeId, double * coord, int dof);

  //! Retrieve the partial stress tensors for the quadrature points in the given element (evaluateInternalForce() must be called prior to getPartialStress()).
  void getPartialStress(std::string blockName, int worksetIndex, int worksetLocalElementId, std::vector< std::vector<RealType>>& partialStressValues);

  //! Accessor for the list of solid elements in the overlap region for optimization-based coupling.
  Teuchos::RCP< std::vector<OBCDataPoint>> getOBCDataPoints(){
    return obcDataPoints;
  }

  //! Retrieve the Epetra_Vector for a given Peridigm data field.
  Teuchos::RCP<const Epetra_Vector> getBlockData(std::string blockName, std::string fieldName);

  //! Sets the output variable list from the user-provided ParameterList.
  void setOutputFields(const Teuchos::ParameterList& params);

  //! Get the list of peridynamics variables that will be written to Exodus.
  std::vector<OutputField> getOutputFields();

  //! Set Dirichlet Fields;
  void setDirichletFields(Teuchos::RCP<Albany::AbstractDiscretization> disc);

  //! Get the STK discretization
  Teuchos::RCP<Albany::STKDiscretization> getSTKDisc(){ return stkDisc; }

private:

  // Peridigm objects
  Teuchos::RCP<PeridigmNS::Discretization> peridynamicDiscretization;
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm;

  Teuchos::RCP<Albany::STKDiscretization> stkDisc;
  Teuchos::RCP<const stk::mesh::MetaData> metaData;
  Teuchos::RCP<const stk::mesh::BulkData> bulkData;

  Teuchos::RCP<Teuchos::ParameterList> peridigmParams;

  Teuchos::RCP<const Teuchos_Comm> teuchosComm;

  bool hasPeridynamics;

  bool enableOptimizationBasedCoupling;
  double obcScaleFactor;

  double previousTime;
  double currentTime;
  double timeStep;

  std::vector<double> previousSolutionPositions;

  std::map<std::string, int> blockNameToBlockId;

  std::vector<OutputField> outputFields;

  std::vector<PartialStressElement> partialStressElements;

  std::vector<int> peridigmNodeGlobalIds;

  std::map<int,int> peridigmGlobalIdToPeridigmLocalId;

  std::vector<int> sphereElementGlobalNodeIds;

  std::map< int, std::vector<int>> worksetLocalIdToGlobalId;

  std::map< int, std::vector<int>> albanyPartialStressElementGlobalIdToPeridigmGlobalIds;

  Teuchos::RCP< std::vector<OBCDataPoint>> obcDataPoints;

  Teuchos::RCP<Epetra_Vector> obcPeridynamicNodeCurrentCoords;

  int cubatureDegree;

  Teuchos::RCP<Tpetra_Vector> albanyOverlapSolutionVector;

  //! Constructor, private to prohibit use.
  PeridigmManager();

  // Private to prohibit use.
  PeridigmManager(const PeridigmManager&);

  // Private to prohibit use.
  PeridigmManager& operator=(const PeridigmManager&);
};

}

#endif // PERIDIGMMANAGER_HPP
