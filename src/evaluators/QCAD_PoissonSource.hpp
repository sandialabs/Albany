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


#ifndef QCAD_POISSONSOURCE_HPP
#define QCAD_POISSONSOURCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

/** 
 * \brief Evaluates Poisson Source Term 
 */
namespace QCAD 
{
	template<typename EvalT, typename Traits>
	class PoissonSource : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> 
  {
	public:
  	typedef typename EvalT::ScalarT ScalarT;
  	typedef typename EvalT::MeshScalarT MeshScalarT;

  	PoissonSource(Teuchos::ParameterList& p);
  
  	void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  	void evaluateFields(typename Traits::EvalData d);
  
  	//! Function to allow parameters to be exposed for embedded analysis
  	ScalarT& getValue(const std::string &n);

        //! Public Universal Constants
        /***** define universal constants as double constants *****/
        static const double kbBoltz; // Boltzmann constant in [eV/K]
        static const double eps0; // vacuum permittivity in [C/(V.cm)]
        static const double eleQ; // electron elemental charge in [C]

	private:

  	//! Reference parameter list generator to check xml input file
  	Teuchos::RCP<const Teuchos::ParameterList>
    		getValidPoissonSourceParameters() const;

        //! evaluateFields functions for different devices (device specified in xml input)
        void evaluateFields_pndiode(typename Traits::EvalData workset);
        void evaluateFields_pmoscap(typename Traits::EvalData workset);
        void evaluateFields_elementblocks(typename Traits::EvalData workset);
        void evaluateFields_default(typename Traits::EvalData workset);

  	//! input
  	std::size_t numQPs;
  	std::size_t numDims;
  	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  	PHX::MDField<ScalarT,Cell,QuadPoint> potential;	// scaled potential, no unit.

  	//! output
  	PHX::MDField<ScalarT,Cell,QuadPoint> poissonSource;
  	PHX::MDField<ScalarT,Cell,QuadPoint> chargeDensity; // in [cm-3]
  	PHX::MDField<ScalarT,Cell,QuadPoint> electronDensity; // in [cm-3]
        PHX::MDField<ScalarT,Cell,QuadPoint> holeDensity; // in [cm-3]
  	PHX::MDField<ScalarT,Cell,QuadPoint> electricPotential;	// phi in [V]

  	//! constant prefactor parameter in source function
  	ScalarT factor;

  	//! Temperature parameter in source function
  	ScalarT temperature; //lattice temperature in [K]

  	//! string variable to differ the various devices implementation
  	std::string device;
        std::string carrierStatistics;
        std::string incompIonization;
    
        //! donor and acceptor concentrations (for element blocks nsilicon & psilicon)
        double dopingDonor;
        double dopingAcceptor;

        //! scaling parameters
        double length_unit_in_m; // length unit for input and output mesh
        double X0;   // length scaling to get to [cm]
        double C0;   // Scaling for conc. [cm^-3]
        ScalarT V0;
        ScalarT Lambda2;

        //! Schrodinger coupling
        bool bSchrodingerInQuantumRegions;
        int  nEigenvectors;
        std::string evecStateRoot;
        std::vector<double> eigenvals;
    
	};
}

#endif
