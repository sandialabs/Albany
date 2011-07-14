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

  //! compute the inverse of the Fermi-Dirac integral of 1/2 order
  ScalarT inverseFDIntOneHalf(const ScalarT x);

  //! built-in potential for MB statistics and complete ionization
  ScalarT potentialForMBComplIon(const ScalarT &Nc, const ScalarT &Nv, 
				 const ScalarT &Eg, const double &Chi,
				 const std::string &dopType, const double &dopingConc);

  //! built-in potential for MB statistics and incomplete ionization
  ScalarT potentialForMBIncomplIon(const ScalarT &Nc, const ScalarT &Nv, 
				   const ScalarT &Eg, const double &Chi, 
				   const std::string &dopType, const double &dopingConc, const double &dopantActE);

  //! built-in potential for FD statistics and complete ionization
  ScalarT potentialForFDComplIon(const ScalarT &Nc, const ScalarT &Nv, 
				 const ScalarT &Eg, const double &Chi, 
				 const std::string &dopType, const double &dopingConc);

  //! built-in potential for zero-K FD statistics and complete ionization
  ScalarT potentialForZeroKFDComplIon(const ScalarT &Nc, const ScalarT &Nv, 
				      const ScalarT &Eg, const double &Chi, 
				      const std::string &dopType, const double &dopingConc);


private:
  ScalarT user_value;    // value entered by user, distinguished from actual DBC value

  std::string material;
  std::string ebName;
  std::string carrierStatistics;
  std::string incompIonization;
  
  double dopingDonor;   // in [cm-3]
  double dopingAcceptor;
  double donorActE;     // (Ec-Ed) where Ed = donor energy level, [eV]
  double acceptorActE;  // (Ea-Ev) where Ea = acceptor energy level, [eV]

  ScalarT temperature;  // [K]
  
  // Since kbT, V0, and qPhiRef are used in all member functions, define them as member variables
  ScalarT kbT;  // [eV]
  ScalarT V0;   // [V]
  ScalarT qPhiRef; //! Constant energy reference for heterogeneous structures,[eV]
  
  //! Material database
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
};

}

#endif
