//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_POISSONSOURCE_HPP
#define QCAD_POISSONSOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"

#include "Albany_Layouts.hpp"

#include "QCAD_MaterialDatabase.hpp"
#include "QCAD_MeshRegion.hpp"
#include "QCAD_EvaluatorTools.hpp"

namespace QCAD {
/** 
 * \brief Evaluates Poisson Source Term 
 */
  template<typename EvalT, typename Traits>
  class PoissonSource : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits>,
  public EvaluatorTools<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    PoissonSource(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  
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
    static const double m0;   // vacuum electron mass in [kg]
    static const double hbar; // reduced planck constant in [J.s]
    static const double pi;   // pi constant (unitless)
    static const double MAX_EXPONENT;  // max. exponent in an exponential function [1]

  private:

    //! Reference parameter list generator to check xml input file
    Teuchos::RCP<const Teuchos::ParameterList>
        getValidPoissonSourceParameters() const;

    //! evaluateFields functions for different device types (device specified in xml input)
    void evaluateFields_elementblocks(typename Traits::EvalData workset);
    void evaluateFields_default(typename Traits::EvalData workset);
    void evaluateFields_moscap1d(typename Traits::EvalData workset);

    //! ----------------- Poisson source setup and fill functions ---------------------

    struct PoissonSourceSetupInfo;
    PoissonSourceSetupInfo source_setup(const std::string& sourceName, const std::string& mtrlCategory,
					const typename Traits::EvalData workset);
    void source_semiclassical(const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
			      const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);
    void source_none         (const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
			      const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);
    void source_quantum      (const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
			      const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);
    void source_coulomb      (const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
			      const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);
    void source_testcoulomb  (const typename Traits::EvalData workset, std::size_t cell, std::size_t qp,
			      const ScalarT& scaleFactor, const PoissonSourceSetupInfo& setup_info);


    //! ----------------- Carrier statistics functions ---------------------

      //! compute the Maxwell-Boltzmann statistics
    inline ScalarT computeMBStat(const ScalarT x);
        
      //! compute the Fermi-Dirac integral of 1/2 order
    inline ScalarT computeFDIntOneHalf(const ScalarT x);
        
      //! compute the 0-K Fermi-Dirac integral
    inline ScalarT computeZeroKFDInt(const ScalarT x);

      //! compute the zero all the time (for insulators)
    inline ScalarT computeZeroStat(const ScalarT x);


    //! ----------------- Activated dopant concentration functions ---------------------
        
      //! return the doping value when incompIonization = False
    inline ScalarT fullDopants(const std::string dopType, const ScalarT &x);
        
      //! compute the ionized dopants when incompIonization = True
    ScalarT ionizedDopants(const std::string dopType, const ScalarT &x);


    //! ----------------- Quantum electron density functions ---------------------
  
#if defined(ALBANY_EPETRA) 
    //! compute the electron density for Poisson-Schrodinger iteration
    ScalarT eDensityForPoissonSchrodinger(typename Traits::EvalData workset, std::size_t cell, 
        std::size_t qp, const ScalarT prevPhi, const bool bUsePredCorr, const double Ef, const double fixedOcc);

    ScalarT eDensityForPoissonCI(typename Traits::EvalData workset, std::size_t cell,
        std::size_t qp, const ScalarT prevPhi, const bool bUsePredCorr, const double Ef, const double fixedOcc);
#endif


    //! ----------------- Point charge functions ---------------------

    //! add point charge contributions to source field
    void source_pointcharges(typename Traits::EvalData workset);

    //! determine whether a point lies within a tetrahedron (used for point charges)
    bool pointIsInTetrahedron(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices);

    //! determine whether a point lies within a hexahedron (used for point charges)
    bool pointIsInHexahedron(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices);

    //! determine whether a point lies within a 2D polygon (used for point charges)
    bool pointIsInPolygon(const MeshScalarT* cellVertices, const MeshScalarT* position, int nVertices);

    //! evaluate determinant of a matrix (used by pointIsInTetrahedra)
    MeshScalarT determinant(const MeshScalarT** mx, int N);

    //! determine whether two points (A and B) lie on the same side of a plane (defined by 3 pts) -- 3D only
    bool sameSideOfPlane(const MeshScalarT* plane0, const MeshScalarT* plane1, const MeshScalarT* plane2, 
			 const MeshScalarT* ptA, const MeshScalarT* ptB);

    //! add cloud charge contributions to source field
    void source_cloudcharges(typename Traits::EvalData workset);

    //! Helper function for point charges
    void update_if_changed(MeshScalarT & oldval, const ScalarT & newval, bool & update_flag) const;

    //! ----------------- Miscellaneous helper functions ---------------------

    //! compute the Fermi-Dirac integral of -1/2 order for calculating electron 
    //! density in the 2D Poisson-Schrondinger loop
    ScalarT computeFDIntMinusOneHalf(const ScalarT x);
    
    //! compute exchange-correlation potential energy within Local Density Approximation
    ScalarT computeVxcLDA(const double& relPerm, const double& effMass, 
        const ScalarT& eDensity); 

    ScalarT getCellScaleFactor(std::size_t cell, const std::vector<bool>& bEBInRegion, ScalarT init_factor);

    ScalarT getReferencePotential(typename Traits::EvalData workset);
    
    //! input
    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numNodes;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVecAtVertices;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    PHX::MDField<ScalarT,Cell,QuadPoint> potential;	// scaled potential (no unit)
    PHX::MDField<ScalarT,Dim> temperatureField; // lattice temperature [K]
    
    //! output
    PHX::MDField<ScalarT,Cell,QuadPoint> poissonSource; // scaled RHS (unitless)
    PHX::MDField<ScalarT,Cell,QuadPoint> chargeDensity; // space charge density in [cm-3]
    PHX::MDField<ScalarT,Cell,QuadPoint> electronDensity; // electron density in [cm-3]
    PHX::MDField<ScalarT,Cell,QuadPoint> artCBDensity; // artificial conduction band density [cm-3]
    PHX::MDField<ScalarT,Cell,QuadPoint> holeDensity;   // electron density in [cm-3]
    PHX::MDField<ScalarT,Cell,QuadPoint> electricPotential;	// phi in [V]
    PHX::MDField<ScalarT,Cell,QuadPoint> ionizedDopant;    // ionized dopants in [cm-3]
    PHX::MDField<ScalarT,Cell,QuadPoint> conductionBand; // conduction band in [eV]
    PHX::MDField<ScalarT,Cell,QuadPoint> valenceBand;   // valence band in [eV]
    PHX::MDField<ScalarT,Cell,QuadPoint> approxQuanEDen;   // approximate quantum electron density [cm-3]

    //! constant prefactor parameter in source function
    ScalarT factor;
    
    //! temperature parameter in source function
    ScalarT temperatureName; //name of temperature field
    
    //! string variable to differ the various devices implementation
    std::string device;

    //! strings specifing the how the source term inside and outside the quantum regions are computed:
    std::string nonQuantumRegionSource;
    std::string quantumRegionSource;
    ScalarT sourceEvecInds[2], prevDensityMixingFactor;
    bool imagPartOfCoulombSrc; // if true, use the imaginary as opposed to real part of coulomb source
    
    //! specify carrier statistics and incomplete ionization
    std::string carrierStatistics;
    std::string incompIonization;
        
    //! donor and acceptor concentrations (for element blocks nsilicon & psilicon)
    double dopingDonor;   // in [cm-3]
    double dopingAcceptor;
        
    //! donor and acceptor activation energy in [eV]
    double donorActE;     // (Ec-Ed) where Ed = donor energy level
    double acceptorActE;  // (Ea-Ev) where Ea = acceptor energy level
        
    //! scaling parameters
    double length_unit_in_m;  // length unit for input and output mesh
    double energy_unit_in_eV; // energy unit for solution, conduction band, etc, but NOT boundary conditions
    //ScalarT C0;  // scaling for conc. [cm^-3]
    //ScalarT Lambda2;  // derived scaling factor (unitless) that appears in the scaled Poisson equation

    //! Mesh Region parameters
    std::vector< Teuchos::RCP<MeshRegion<EvalT, Traits> > > meshRegionList;
    std::vector< ScalarT > meshRegionFactors;

    //! Point Charge parameters
    struct PointCharge { MeshScalarT position[3]; ScalarT position_param[3]; ScalarT charge; int iWorkset, iCell; };
    std::vector< PointCharge > pointCharges;
    std::size_t numWorksetsScannedForPtCharges;
    
    //! Cloud Charge parameters
    struct CloudCharge { ScalarT position[3]; ScalarT amplitude, width, cutoff;};
    std::vector< CloudCharge > cloudCharges;
    
    //! Schrodinger coupling
    bool bUsePredictorCorrector;
    bool bIncludeVxc; 
    bool bRealEigenvectors;
    int  nEigenvectors;
    double fixedQuantumOcc;
    std::vector< PHX::MDField<ScalarT,Cell,QuadPoint> > eigenvector_Re;
    std::vector< PHX::MDField<ScalarT,Cell,QuadPoint> > eigenvector_Im;
    
    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! Material database parameter values
    std::map<std::string, ScalarT > materialParams;
    
    //! specific parameters for "1D MOSCapacitor"
    double oxideWidth;
    double siliconWidth; 
    
    //! Map element block and nodeset names to their associated DBC values
    std::map<std::string, double> mapDBCValue_eb; 
    std::map<std::string, double> mapDBCValue_ns; 


    struct PoissonSourceSetupInfo
    {
      ScalarT qPhiRef; // energy reference for heterogeneous structures, in [eV]
      ScalarT Lambda2; // derived scaling factor
      ScalarT V0;      // kb*T in desired energy unit ( or kb*T/q in desired voltage unit), [myV]
      ScalarT kbT;     // in [eV]
      double Chi;      // Electron affinity (in semiconductors & insulators) or Work function (in metals) in [eV]
      
      //Semiconductors
      double fermiE; // Fermi energy, [myV]
      ScalarT eArgOffset, hArgOffset;  //! argument offset in calculating electron and hole density [unitless]
      
      
      //! strong temperature-dependent material parameters
      ScalarT Nc;  // conduction band effective DOS in [cm-3]
      ScalarT Nv;  // valence band effective DOS in [cm-3]
      ScalarT Eg;  // band gap at T [K] in [eV]
      
      //! Activated dopants / Fixed constant charge
      std::string fixedChargeType;
      ScalarT dopingConc, fixedChargeConc;  // [cm-3]
      ScalarT inArg;
      
      //for predictor corrector
      Albany::MDArray prevPhiArray;

      //for damping/mixing
      double prevDensityFactor;
      Albany::MDArray prevDensityArray;
      
      //for coulomb
      int sourceEvec1, sourceEvec2;
      ScalarT coulombPrefactor;
      
      // for exchange-correlation
      double averagedEffMass;
      double relPerm;
      
      //! function pointer to carrier statistics member function
      ScalarT (QCAD::PoissonSource<EvalT,Traits>::*carrStat) (const ScalarT);
      
      //! function pointer to ionized dopants member function
      ScalarT (QCAD::PoissonSource<EvalT,Traits>::*ionDopant) (const std::string, const ScalarT&); 
      
      //! function pointer to quantum electron density member function
      ScalarT (QCAD::PoissonSource<EvalT,Traits>::*quantum_edensity_fn) 
      (typename Traits::EvalData, std::size_t, std::size_t, const ScalarT, const bool, const double, const double);
    };

  };

}

#endif
