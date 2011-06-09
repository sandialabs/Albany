/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef QCAD_POISSONDIRICHLET_HPP
#define QCAD_POISSONDIRICHLET_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

#include "QCAD_MaterialDatabase.hpp"

/** \brief Dirichlet Evaluator for QCAD Poisson Problem
*/

namespace QCAD {

template<typename EvalT, typename Traits>
class PoissonDirichlet
  : public PHAL::Dirichlet<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;

  PoissonDirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
  virtual ScalarT& getValue(const std::string &n) { return user_value; }

protected:
  //! Compute the (scaled) potential offset due to a doped element block
  ScalarT ComputeOffsetDueToDoping(const std::string ebName, ScalarT phiForIncompleteIon);

  //! compute the inverse of Maxwell-Boltzmann statistics
  inline ScalarT invComputeMBStat(const ScalarT x);
        
  //! compute the inverse of the Fermi-Dirac integral of 1/2 order
  inline ScalarT invComputeFDIntOneHalf(const ScalarT x);
        
  //! compute the inverse of the 0-K Fermi-Dirac integral
  inline ScalarT invComputeZeroKFDInt(const ScalarT x);
        
  //! return the doping value when incompIonization = False
  inline ScalarT fullDopants(const std::string dopType, const ScalarT &x);
        
  //! compute the ionized dopants when incompIonization = True
  ScalarT ionizedDopants(const std::string dopType, const ScalarT &x);


protected:
  ScalarT user_value;    // value entered by user, distinguished from actual DBC value

  std::string ebSetByUser;
  std::string ebOfDOF;
  double offsetDueToAffinity;

  std::string carrierStatistics;
  std::string incompIonization;
  double dopingDonor;   // in [cm-3]
  double dopingAcceptor;
  double donorActE;     // (Ec-Ed) where Ed = donor energy level
  double acceptorActE;  // (Ea-Ev) where Ea = acceptor energy level

  ScalarT temperature;  // [K]

  //! Material database
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
};

}

#endif
