//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_FerroicModel_hpp)
#define LCM_FerroicModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid2_MiniTensor.h>

#include "Sacado.hpp"

namespace LCM
{

//! \brief Ferroic model for electromechanics
template<typename EvalT, typename Traits>
class FerroicModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType,ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  FerroicModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~FerroicModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  //Kokkos
  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

private:
  
  ///
  /// Private methods
  ///
  template<typename T>
  void computeState(Teuchos::Array<T>& fractions,
                    Intrepid2::Tensor<ScalarT>& x, 
                    Intrepid2::Tensor<T>& X, 
                    Intrepid2::Tensor<T>& linear_x,
                    Intrepid2::Vector<ScalarT>& E, 
                    Intrepid2::Vector<T>& D, 
                    Intrepid2::Vector<T>& linear_D);

  void findActiveTransitions(Intrepid2::FieldContainer_Kokkos<int, PHX::Layout, PHX::Device>& transitionMap,
                             Teuchos::Array<ScalarT>& fractions,
                             Intrepid2::Tensor<ScalarT>& X, Intrepid2::Tensor<ScalarT>& linear_x,
                             Intrepid2::Vector<ScalarT>& E, Intrepid2::Vector<ScalarT>& linear_D);

  void findEquilibriumState(Intrepid2::FieldContainer_Kokkos<int, PHX::Layout, PHX::Device>& transitionMap,
                            Teuchos::Array<ScalarT>& oldfractions,
                            Teuchos::Array<ScalarT>& newfractions,
                            Intrepid2::Tensor<ScalarT>& x, Intrepid2::Vector<ScalarT>& E);

  bool converged(std::vector<ScalarT>& R, int iteration, ScalarT& initNorm);


  void computeResidualandJacobian(Intrepid2::FieldContainer_Kokkos<int, PHX::Layout, PHX::Device> transitionMap,
                                  Intrepid2::Tensor<ScalarT>& x, Intrepid2::Vector<ScalarT>& E,
                                  Teuchos::Array<ScalarT>& fractions,
                                  std::vector<ScalarT>& X, std::vector<ScalarT>& R,
                                  std::vector<ScalarT>& dRdX);

  ///
  /// Private to prohibit copying
  ///
  FerroicModel(const FerroicModel&);

  ///
  /// Private to prohibit copying
  ///
  FerroicModel& operator=(const FerroicModel&);

  ///
  /// material parameters
  ///
  Intrepid2::Tensor<RealType> R;

//  Intrepid2::Tensor4<RealType> C;
//  Intrepid2::Tensor3<RealType> h;
//  Intrepid2::Tensor<RealType> beta;


  ///
  /// INDEPENDENT FIELD NAMES
  ///
  std::string strainName;
  std::string efieldName;
 
  ///
  /// EVALUATED FIELD NAMES
  ///
  std::string stressName;
  std::string edispName;

  ///
  /// STATE VARIABLE NAMES
  ///
  Teuchos::Array<std::string> binNames;

  class CrystalPhase {
   public:
    CrystalPhase(Intrepid2::Tensor<RealType>& R, Teuchos::ParameterList& p);

    Intrepid2::Tensor4<RealType> C;
    Intrepid2::Tensor3<RealType> h;
    Intrepid2::Tensor<RealType> beta;

    // Rhombohedral ?
    RealType C11, C33, C12, C23, C44, C66;
    RealType h31, h33, h15, E11, E33;

    // generalize for least symmetry
    // error checking based on specified symmetry?

  };
  
  class CrystalVariant {
   public:
    CrystalVariant(Teuchos::Array<Teuchos::RCP<CrystalPhase>>& phases, Teuchos::ParameterList& p);
    Intrepid2::Tensor4<RealType> C;
    Intrepid2::Tensor3<RealType> h;
    Intrepid2::Tensor<RealType> beta;
    Intrepid2::Tensor<RealType> R;
    Intrepid2::Tensor<RealType> spontStrain;
    Intrepid2::Vector<RealType> spontEDisp;
  };

  class Transition {
   public:
    Transition(Teuchos::RCP<CrystalVariant> from, Teuchos::RCP<CrystalVariant> to);
    Teuchos::RCP<CrystalVariant> fromVariant;
    Teuchos::RCP<CrystalVariant> toVariant;
    Intrepid2::Tensor<RealType> transStrain;
    Intrepid2::Vector<RealType> transEDisp;
  };

  Teuchos::Array<RealType> initialBinFractions;
  Teuchos::Array<Teuchos::RCP<CrystalPhase>> crystalPhases;
  Teuchos::Array<Teuchos::RCP<CrystalVariant>> crystalVariants;
  Teuchos::Array<Teuchos::RCP<Transition>> transitions;
  Teuchos::Array<ScalarT> tBarrier;
  RealType alphaParam, gammaParam;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> aMatrix;

};

void parseBasis(const Teuchos::ParameterList& pBasis, Intrepid2::Tensor<RealType>& R);

void changeBasis(Intrepid2::Tensor4<RealType>& inMatlBasis, 
                 const Intrepid2::Tensor4<RealType>& inGlobalBasis,
                 const Intrepid2::Tensor<RealType>& Basis);
void changeBasis(Intrepid2::Tensor3<RealType>& inMatlBasis, 
                 const Intrepid2::Tensor3<RealType>& inGlobalBasis,
                 const Intrepid2::Tensor<RealType>& Basis);
void changeBasis(Intrepid2::Tensor<RealType>& inMatlBasis, 
                 const Intrepid2::Tensor<RealType>& inGlobalBasis,
                 const Intrepid2::Tensor<RealType>& Basis);

}

#endif
