//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_SYNTETIC_TEST_BC_HPP
#define LANDICE_STOKES_FO_SYNTETIC_TEST_BC_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, typename BetaScalarT>
class StokesFOSynteticTestBC : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  StokesFOSynteticTestBC (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  enum class BCType {
    CONSTANT,
    EXPTRIG,
    ISMIP_HOM_TEST_C,
    ISMIP_HOM_TEST_D,
    CIRCULAR_SHELF,
    CONFINED_SHELF,
    XZ_MMS
  };

  // Input:
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint,VecDim>    u;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim>   qp_coords;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim>   side_normals;
  PHX::MDField<const RealType,Cell,Side,Node,QuadPoint>     BF;
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>       w_measure;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim>                    residual;

  std::vector<std::vector<int> >  sideNodes;
  std::string                     ssName;

  Kokkos::DynRankView<ScalarT, PHX::Device>           qp_temp_buffer;

  int numSideNodes;
  int numSideQPs;
  int sideDim;
  int vecDimFO;

  // Parameters used by the bc's (not necessarily by all of them).
  double alpha;
  double beta;
  double beta1;
  double beta2;
  double L;
  double n;
  Teuchos::Array<int> components;

  BCType  bc_type;
};

} // Namespace LandIce

#endif // LANDICE_STOKES_FO_SYNTETIC_TEST_BC_HPP
