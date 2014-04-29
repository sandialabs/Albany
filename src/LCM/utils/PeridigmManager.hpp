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

  //! Singleton.
  static PeridigmManager & self();

  //! Returns initialization flag.
  bool isInitialized();

  //! Add global element ids to the Peridigm id list
  void addGlobalElementIds(const std::vector<int>& ids);

  //! Instantiate the Peridigm object
  void initialize(const Teuchos::RCP<Teuchos::ParameterList>& params,
		  Teuchos::RCP<Albany::AbstractDiscretization> disc);

  //! Load the current displacements from the Albany solution vector into the Peridigm manager.
  void setDisplacements(const Epetra_Vector& x);

  //! Evaluate the peridynamic internal force
  void evaluateInternalForce();

  //! Retrieve the force for the given global degree of freedom (evaluateInternalForce() must be called prior to getForce()).
  double getForce(int globalId, int dof);

private:

#ifdef ALBANY_PERIDIGM
  // Peridigm objects
  Teuchos::RCP<PeridigmNS::Discretization> peridynamicDiscretization;
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm;
#endif

  bool peridigmIsInitialized;

  Teuchos::RCP<Teuchos::ParameterList> peridigmParams;

  Teuchos::RCP<Epetra_Comm> epetraComm;

  std::vector<int> myGlobalElements;

  //! Constructor, private to prohibit use.
  PeridigmManager();

  // Private to prohibit use.
  PeridigmManager(const PeridigmManager&);

  // Private to prohibit use.
  PeridigmManager& operator=(const PeridigmManager&);
};

}

#endif // PERIDIGMMANAGER_HPP
