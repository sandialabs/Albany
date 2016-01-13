//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_VELRESID_HPP
#define AERAS_HYDROSTATIC_VELRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Hydrostatic equation Residual for atmospheric modeling

    This evaluator computes the residual of the Hydrostatic equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class Hydrostatic_VelResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Hydrostatic_VelResid(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<Intrepid2::Cubature<RealType> > cubature;
  Intrepid2::FieldContainer<RealType>    refPoints;
  Intrepid2::FieldContainer<RealType>    refWeights;
  Intrepid2::FieldContainer<RealType>    grad_at_cub_points;

  // vorticity only returns the component in the radial direction
  //void get_vorticity(const Intrepid2::FieldContainer<ScalarT>  & fieldAtNodes,
  //    std::size_t cell, std::size_t level, Intrepid2::FieldContainer<ScalarT>  & curl);

  void get_coriolis(std::size_t cell, Intrepid2::FieldContainer<ScalarT>  & coriolis);

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>         wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>     GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>     wGradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim,Dim> wGradGradBF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>          sphere_coord;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim>      jacobian;
  PHX::MDField<MeshScalarT,Cell,QuadPoint>              jacobian_det;

  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  keGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  PhiGrad;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  etadotdVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  pGrad;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  Velx;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>       VelxNode;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  VelxDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  DVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      LaplaceVelx;
  PHX::MDField<ScalarT,Cell,Node,Level>      density;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      vorticity;


  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Residual;

  const double viscosity;
  const double hyperviscosity;
  const double AlphaAngle;
  const double Omega;

  const int numNodes;
  const int numQPs;
  const int numDims;
  const int numLevels;
};
}

#endif
