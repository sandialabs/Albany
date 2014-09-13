//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Porosity_hpp)
#define LCM_Porosity_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace LCM {
  ///
  /// \brief Evaluates porosity, either as a constant or a truncated
  /// KL expansion.
  ///
  /// Porosity update is the most important part for the poromechanics
  /// formulation. All poroelasticity parameters (Biot Coefficient,
  /// Biot modulus, permeability, and consistent tangential tensor)
  /// depend on porosity. The definition we used here is from
  /// Coussy's poromechanics p.85.
  ///
  template<typename EvalT, typename Traits>
  class Porosity :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Constructor
    ///    
    Porosity(Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl);
  
    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);
  
    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);
  
    ///
    /// Sacado method to access parameters
    ///
    ScalarT& getValue(const std::string &n);

  private:

    ///
    /// Number of integration points
    ///
    std::size_t numQPs;

    ///
    /// Number of problem dimensions
    ///
    std::size_t numDims;

    ///
    /// Container for coordinates
    ///
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

    ///
    /// Container for porosity
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> porosity;

    ///
    /// Is porosity constant, or random field
    ///
    bool is_constant;

    ///
    /// Constant value
    ///
    ScalarT constant_value;

    ///
    /// Optional dependence on strain and porePressure
    ///
    /// porosity holds linear relation to volumetric strain
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> strain;

    ///
    /// Optional dependence on det(F)
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> J;
    
    ///
    /// flag to indicated usage in poroelastic context
    ///
    bool isPoroElastic;

    ///
    /// flag
    ///
    bool isCompressibleSolidPhase;

    ///
    /// flag
    ///
    bool isCompressibleFluidPhase;

    ///
    /// initial state
    ///
    ScalarT initialPorosityValue;

    ///
    /// For compressible grain
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> biotCoefficient;

    ///
    /// For compressible grain
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> porePressure;

    ///
    /// For compressible grain
    ///
    ScalarT GrainBulkModulus;

    ///
    /// For THM porous media
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;

    ///
   /// For THM porous media
   ///
   PHX::MDField<ScalarT,Cell,QuadPoint> skeletonThermalExpansion;

   ///
  /// For THM porous media
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint> refTemperature;

    ///
    /// Exponential random field
    ///
    Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;

    ///
    /// Values of the random variables
    ///
    Teuchos::Array<ScalarT> rv;

    ///
    /// Strain flag
    ///
    bool hasStrain;

    ///
    /// J flag
    ///
    bool hasJ;

    ///
    /// J flag
    ///
    bool hasTemp;
  };
}

#endif
