//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
          const Teuchos::RCP<std::vector<double> >& pathLen_, int nGFPts_,  
				  double ptSpacing_, double effMass_, 
				  const Teuchos::RCP<const Epetra_Comm>& Comm_,
				  const std::string& outputFilename,
				  bool bNeumannBC_ = true);
    ~GreensFunctionTunnelingSolver();

    // returns current in units of Amps at a given Vds (in Volts)
    double computeCurrent(double Vds, double kbT, double Ecutoff_offset_from_Emax, bool bUseAnasazi);
    
    void computeCurrentRange(const std::vector<double> Vds, double kbT, double Ecutoff_offset_from_Emax,
			     std::vector<double>& resultingCurrent, bool bUseAnasazi);

  private:
    double f0(double x) const;

    bool doMatrixDiag_Anasazi(double Vds, std::vector<double>& Ec, double Ecutoff,
		      std::vector<double>& evals,
		      Teuchos::RCP<Epetra_MultiVector>& evecs);
    bool doMatrixDiag_tql2(double Vds, std::vector<double>& Ec, double Ecutoff,
			   std::vector<double>& evals, 
			   std::vector<double>& evecs);
    int tql2(int n, int max_iter, std::vector<double>& d, std::vector<double>& e, 
	     std::vector<double>& z, int& ierr);
	  void sortTql2PartialResults(int n, std::vector<double>& d, std::vector<double>& z); 


    // y2 must be passed by reference since its values are changed in the routine
    void prepSplineInterp(const std::vector<double>& x, const std::vector<double>& y,  
          std::vector<double>& y2, const int& n); 

    // klo and khi must be passed by reference since their values are changed in the routine
    double execSplineInterp(const std::vector<double>& xa, const std::vector<double>& ya, 
          const std::vector<double>& y2a, const int& n, const double& x); 
   
  private:
    Teuchos::RCP<const Epetra_Comm> Comm;
    Teuchos::RCP<Epetra_Map> Map;
    Teuchos::RCP<std::vector<double> > EcValues;
    
    double ptSpacing, effMass, t0;
    bool bNeumannBC;
    int nGFPts; 
    
    std::vector<double> matlabEvals; 
  };

}

#endif
