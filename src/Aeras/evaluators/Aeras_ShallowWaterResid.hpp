//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERRESID_HPP
#define AERAS_SHALLOWWATERRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace Aeras {
/** \brief ShallowWater equation Residual for atmospheric modeling

    This evaluator computes the residual of the ShallowWater equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class ShallowWaterResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ShallowWaterResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> U;  //vecDim works but its really Dim+1
  PHX::MDField<ScalarT,Cell,Node,VecDim> UNodal;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> UDot;

  PHX::MDField<ScalarT,Cell,QuadPoint> mountainHeight;

  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det;
  Intrepid::FieldContainer<RealType>    grad_at_cub_points;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;


  bool usePrescribedVelocity;
  bool ibpGradH;

  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Intrepid::FieldContainer<RealType>    refPoints;
  Intrepid::FieldContainer<RealType>    refWeights;
  Intrepid::FieldContainer<MeshScalarT>  nodal_jacobian;
  Intrepid::FieldContainer<MeshScalarT>  nodal_inv_jacobian;
  Intrepid::FieldContainer<MeshScalarT>  nodal_det_j;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>   sphere_coord;

  ScalarT gravity; // gravity parameter -- Sacado-ized for sensitivities
  ScalarT Omega;   //rotation of earth  -- Sacado-ized for sensitivities

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;
  std::size_t spatialDim;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  void divergence(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
      std::size_t cell, Intrepid::FieldContainer<ScalarT>  & div);

  //gradient returns vector in physical basis
  void gradient(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
      std::size_t cell, Intrepid::FieldContainer<ScalarT>  & gradField);

  // curl only returns the component in the radial direction
  void curl(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
      std::size_t cell, Intrepid::FieldContainer<ScalarT>  & curl);

  void fill_nodal_metrics(std::size_t cell);

  void get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis);

};
  const int qpToNodeMap[9] = {0, 4, 1,
      7, 8, 5,
      3, 6, 2 };
const int nodeToQPMap[9]  = {0, 2, 8, 6,
    1, 5, 7, 3,
    4 };


}

#endif
