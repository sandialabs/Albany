/*! \file PeridigmManager.hpp */

#ifndef PERIDIGMMANAGER_HPP
#define PERIDIGMMANAGER_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"

#ifdef ALBANY_PERIDIGM
#include <Peridigm.hpp>
#include <Peridigm_AlbanyDiscretization.hpp>
#endif

namespace LCM {

class PeridigmManager {

public:

  struct OutputField {
    std::string albanyName;
    std::string peridigmName;
    std::string lengthName;
    std::string relation;
    int length;
    bool operator==(const OutputField &rhs){
      if(rhs.albanyName != albanyName ||
         rhs.peridigmName != peridigmName ||
         rhs.lengthName != lengthName ||
         rhs.relation != relation)
        return false;
      return true;
    }
  } ;

  //! Singleton.
  static PeridigmManager & self();

  //! Instantiate the Peridigm object
  void initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  Teuchos::RCP<Albany::AbstractDiscretization> disc);

  //! Load the current time and displacement from Albany into the Peridigm manager.
  void setCurrentTimeAndDisplacement(double time, const Epetra_Vector& albanySolutionVector);

  //! Evaluate the peridynamic internal force
  void evaluateInternalForce();

  //! Update the state within Peridigm following a successful load step.
  void updateState();

  //! Retrieve the force for the given global degree of freedom (evaluateInternalForce() must be called prior to getForce()).
  double getForce(int globalId, int dof);

  //! Retrieve the Epetra_Vector for a given Peridigm data field.
  Teuchos::RCP<const Epetra_Vector> getBlockData(std::string blockName, std::string fieldName);

  //! Sets the output variable list from the user-provided ParameterList.
  void setOutputFields(const Teuchos::ParameterList& params);

  //! Get the list of peridynamics variables that will be written to Exodus.
  std::vector<OutputField> getOutputFields();

private:

#ifdef ALBANY_PERIDIGM
  // Peridigm objects
  Teuchos::RCP<PeridigmNS::Discretization> peridynamicDiscretization;
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm;
#endif

  Teuchos::RCP<Teuchos::ParameterList> peridigmParams;

  Teuchos::RCP<Epetra_Comm> epetraComm;

  bool hasPeridynamics;

  double previousTime;
  double currentTime;
  double timeStep;

  Teuchos::RCP<Epetra_Vector> previousSolutionPositions;

  std::vector<OutputField> outputFields;

  //! Constructor, private to prohibit use.
  PeridigmManager();

  // Private to prohibit use.
  PeridigmManager(const PeridigmManager&);

  // Private to prohibit use.
  PeridigmManager& operator=(const PeridigmManager&);
};

}

#endif // PERIDIGMMANAGER_HPP
