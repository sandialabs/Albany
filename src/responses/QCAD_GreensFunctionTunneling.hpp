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


#ifndef QCAD_GREENSFUNCTIONTUNNELING_HPP
#define QCAD_GREENSFUNCTIONTUNNELING_HPP

#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "AnasaziEpetraAdapter.hpp"

namespace QCAD {

  class GreensFunctionTunnelingSolver
  {
  public:

    // EcValues are in units of eV
    // effMass is in units of m_0 (electron rest mass)
    // ptSpacing is in units of microns (um)
    GreensFunctionTunnelingSolver(const Teuchos::RCP<std::vector<double> >& EcValues_, 
				  double ptSpacing, double effMass, 
				  const Teuchos::RCP<Epetra_Comm>& Comm_);
    ~GreensFunctionTunnelingSolver();

    // returns current in units of Amps at a given Vds (in Volts)
    double computeCurrent(double Vds, double kbT);
    
    void computeCurrentRange(const std::vector<double> Vds, double kbT, std::vector<double>& resultingCurrent);

  private:
    double f0(double x) const;

    bool doMatrixDiag(double Vds, std::vector<double>& Ec, double Emax,
		      std::vector<Anasazi::Value<double> >& evals,
		      Teuchos::RCP<Epetra_MultiVector>& evecs);

  private:
    Teuchos::RCP<Epetra_Comm> Comm;
    Teuchos::RCP<Epetra_Map> Map;
    Teuchos::RCP<std::vector<double> >EcValues;
    double ptSpacing, mass, t0;
  };

}

#endif
