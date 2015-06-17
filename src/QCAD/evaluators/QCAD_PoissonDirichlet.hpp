//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_POISSONDIRICHLET_HPP
#define QCAD_POISSONDIRICHLET_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace QCAD {

/** \brief Dirichlet Evaluator for QCAD Poisson Problem
*/

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

  //! built-in potential for Schottky barrier
  ScalarT potentialForSchottkyBarrier(const ScalarT &Eg, const double &Chi, 
				  const std::string &dopType);

private:
  ScalarT user_value;    // value entered by user, distinguished from actual DBC value

  std::string material;
  std::string ebName;
  std::string carrierStatistics;
  std::string incompIonization;
  std::string nodeSetName; 
  std::string contactType; 
  
  double dopingDonor;   // in [cm-3]
  double dopingAcceptor;
  double donorActE;     // (Ec-Ed) where Ed = donor energy level, [eV]
  double acceptorActE;  // (Ea-Ev) where Ea = acceptor energy level, [eV]
  double sbHeight;      // Schottky barrier height in [eV]

  ScalarT temperature;  // [K]
  
  // Since kbT, V0, and qPhiRef are used in all member functions, define them as member variables
  ScalarT kbT;  // [eV]
  ScalarT V0;   // [V]
  ScalarT qPhiRef; //! Constant energy reference for heterogeneous structures,[eV]
  double energy_unit_in_eV; // to convert eV -> unit of solution (Phi)
  
  //! Material database
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
};

}

#endif
