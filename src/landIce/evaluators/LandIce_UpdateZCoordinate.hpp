//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_UPDATE_Z_COORDINATE_HPP
#define LANDICE_UPDATE_Z_COORDINATE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 

#include "Albany_SacadoTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, typename ScalarT>
class UpdateZCoordinateMovingTopBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  UpdateZCoordinateMovingTopBase(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData /* d */,
                             PHX::FieldManager<Traits>& /* vm */);

  void evaluateFields(typename Traits::EvalData d);

private:

  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node,Dim>   coordVecIn;
  PHX::MDField<const MeshScalarT, Cell, Node>       bedTopo;
  PHX::MDField<const ScalarT, Cell, Node>           H;
  PHX::MDField<const RealType, Cell, Node>          H0;
  PHX::MDField<const ScalarT, Cell, Node>           dH;

  // Output:
  PHX::MDField<ScalarT, Cell, Node>           topSurface;
  PHX::MDField<MeshScalarT, Cell, Node, Dim>  coordVecOut;

  bool haveThickness;
  double minH, rho_i, rho_w;
  int numDims, numNodes;
};

// Shortcut name
template<typename EvalT, typename Traits>
using UpdateZCoordinateMovingTop = UpdateZCoordinateMovingTopBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
class UpdateZCoordinateMovingBed : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  UpdateZCoordinateMovingBed(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData /* d */,
                             PHX::FieldManager<Traits>& /* vm */);

  void evaluateFields(typename Traits::EvalData d);

private:

  using ParamScalarT = typename EvalT::ParamScalarT;
  using MeshScalarT  = typename EvalT::MeshScalarT;

  using ScalarOutT = typename Albany::StrongestScalarType<MeshScalarT,ParamScalarT>::type;

  // Input:
  PHX::MDField<const MeshScalarT,  Cell, Node,Dim>   coordVecIn;
  PHX::MDField<const RealType,     Cell, Node>       bedTopo;
  PHX::MDField<const RealType,     Cell, Node>       topSurface;
  PHX::MDField<const ParamScalarT, Cell, Node>       H;

  // Output:
  PHX::MDField<ScalarOutT, Cell, Node, Dim>  coordVecOut;
  PHX::MDField<ScalarOutT, Cell, Node>       topSurfaceOut;
  PHX::MDField<ScalarOutT, Cell, Node>       bedTopoOut;

  double minH, rho_i, rho_w;
  int numDims, numNodes;
};


template<typename EvalT, typename Traits>
class UpdateZCoordinateGivenTopAndBedSurfaces : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  UpdateZCoordinateGivenTopAndBedSurfaces(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData /* d */,
                             PHX::FieldManager<Traits>& /* vm */);

  void evaluateFields(typename Traits::EvalData d);

private:

  using MeshScalarT = typename EvalT::MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node,Dim> coordVecIn;
  PHX::MDField<const MeshScalarT, Cell, Node>     topSurfIn;
  PHX::MDField<const MeshScalarT, Cell, Node>     bedTopoIn;

  // Output:
  PHX::MDField<MeshScalarT, Cell, Node>  H;
  PHX::MDField<MeshScalarT, Cell, Node, Dim>  coordVecOut;

  // InOut:
  // output if isBedSurfParam == true
  PHX::MDField<MeshScalarT, Cell, Node>        bedTopo;

  // output if isBedSurfParam == false
  PHX::MDField<MeshScalarT, Cell, Node>        topSurf;

  bool isTopSurfParam, isBedTopoParam;
  double minH, rho_i, rho_w;
  int numDims, numNodes;
};

} // namespace LandIce

#endif // LANDICE_UPDATE_Z_COORDINATE_HPP
