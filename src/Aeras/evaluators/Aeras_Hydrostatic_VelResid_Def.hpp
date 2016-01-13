//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
Hydrostatic_VelResid<EvalT, Traits>::
Hydrostatic_VelResid(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF         (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"),               dl->node_qp_gradient),
  wGradBF     (p.get<std::string> ("Weighted Gradient BF Name"),        dl->node_qp_gradient),
  wGradGradBF (p.isParameter("Hydrostatic Problem") &&  
               p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->isParameter("HyperViscosity") ?
               p.get<std::string> ("Weighted Gradient Gradient BF Name") : "None", dl->node_qp_tensor) ,
  keGrad      (p.get<std::string> ("Gradient QP Kinetic Energy"),       dl->qp_gradient_level),
  PhiGrad     (p.get<std::string> ("Gradient QP GeoPotential"),         dl->qp_gradient_level),
  etadotdVelx (p.get<std::string> ("EtaDotdVelx"),                      dl->qp_vector_level),
  pGrad       (p.get<std::string> ("Gradient QP Pressure"),             dl->qp_gradient_level),
  VelxNode    (p.get<std::string> ("Velx"),                             dl->node_vector_level),
  Velx        (p.get<std::string> ("QP Velx"),                          dl->qp_vector_level),
  VelxDot     (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_vector_level),
  DVelx       (p.get<std::string> ("D Vel Name"),                       dl->qp_vector_level),
  LaplaceVelx (p.isParameter("Hydrostatic Problem") &&
                p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->isParameter("HyperViscosity") ?
                p.get<std::string> ("Laplace Vel Name") : "None",dl->qp_scalar_level),
  density     (p.get<std::string> ("QP Density"),                       dl->qp_scalar_level),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  vorticity    (p.get<std::string>  ("QP Vorticity"), dl->qp_scalar_level ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),

  Residual    (p.get<std::string> ("Residual Name"),                    dl->node_vector_level),

  viscosity   (p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Viscosity", 0.0)),
  hyperviscosity (p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("HyperViscosity", 0.0)),
  AlphaAngle (p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Rotation Angle", 0.0)),
  //AlphaAngle (p.isParameter("XZHydrostatic Problem") ? 
  //              p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Rotation Angle", 0.0):
  //              p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Rotation Angle", 0.0)),
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.)),

  numNodes    ( dl->node_scalar             ->dimension(1)),
  numQPs      ( dl->node_qp_scalar          ->dimension(2)),
  numDims     ( dl->node_qp_gradient        ->dimension(3)),
  numLevels   ( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(keGrad);
  this->addDependentField(PhiGrad);
  this->addDependentField(density);
  this->addDependentField(etadotdVelx);
  this->addDependentField(DVelx);
  this->addDependentField(pGrad);
  this->addDependentField(VelxNode);
  this->addDependentField(Velx);
  this->addDependentField(VelxDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(sphere_coord);
  this->addDependentField(vorticity);
  this->addDependentField(jacobian);
  this->addDependentField(jacobian_det);
  if (hyperviscosity) this->addDependentField(LaplaceVelx);
  if (hyperviscosity) this->addDependentField(wGradGradBF);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::Hydrostatic_VelResid" );

  //refWeights        .resize               (numQPs);
  //grad_at_cub_points.resize     (numNodes, numQPs, 2);
  //refPoints         .resize               (numQPs, 2);

  //cubature->getCubature(refPoints, refWeights);
  //intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_VelResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(keGrad     , fm);
  this->utils.setFieldData(PhiGrad    , fm);
  this->utils.setFieldData(density    , fm);
  this->utils.setFieldData(etadotdVelx, fm);
  this->utils.setFieldData(DVelx      , fm);
  this->utils.setFieldData(pGrad      , fm);
  this->utils.setFieldData(VelxNode   , fm);
  this->utils.setFieldData(Velx       , fm);
  this->utils.setFieldData(VelxDot    , fm);
  this->utils.setFieldData(wBF        , fm);
  this->utils.setFieldData(wGradBF    , fm);
  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(vorticity  , fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_det, fm);
  if (hyperviscosity) this->utils.setFieldData(LaplaceVelx, fm);
  if (hyperviscosity) this->utils.setFieldData(wGradGradBF, fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_VelResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  double n_coeff = workset.n_coeff;
  obtainLaplaceOp = (n_coeff == 1) ? true : false;

  PHAL::set(Residual, 0.0);
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  coriolis(numQPs);
  //Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  vorticity(numQPs);


  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {

        get_coriolis(cell, coriolis);
        //get_vorticity(VelxNode, cell, level, vorticity);

        for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim < numDims; ++dim) {
            Residual(cell,node,level,dim) += ( keGrad(cell,qp,level,dim) + PhiGrad(cell,qp,level,dim) )*wBF(cell,node,qp);
            Residual(cell,node,level,dim) += ( pGrad (cell,qp,level,dim)/density(cell,qp,level) )      *wBF(cell,node,qp);
            Residual(cell,node,level,dim) +=   etadotdVelx(cell,qp,level,dim)                          *wBF(cell,node,qp);
            Residual(cell,node,level,dim) +=   VelxDot(cell,qp,level,dim)                              *wBF(cell,node,qp);
            Residual(cell,node,level,dim) +=   viscosity * DVelx(cell,qp,level,dim)                    *wGradBF(cell,node,qp,dim);
            if (hyperviscosity) 
              Residual(cell,node,level,dim) -= hyperviscosity * LaplaceVelx(cell,qp,level) * wGradGradBF(cell,node,qp,dim,dim);
          }
          Residual(cell,node,level,0) -= (vorticity(cell,qp,level) + coriolis(qp))*Velx(cell,qp,level,1)*wBF(cell,node,qp);
          Residual(cell,node,level,1) += (vorticity(cell,qp,level) + coriolis(qp))*Velx(cell,qp,level,0)*wBF(cell,node,qp);
        }
      }
    }
  }
}

template<typename EvalT,typename Traits>
void
Hydrostatic_VelResid<EvalT,Traits>::get_coriolis(std::size_t cell, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & coriolis) {

  coriolis.initialize();
  double alpha = AlphaAngle; 

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
  }

}

// *********************************************************************

//template<typename EvalT,typename Traits>
//void
//Hydrostatic_VelResid<EvalT,Traits>::get_vorticity(const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & nodalVector,
//    std::size_t cell, std::size_t level, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & curl) {
//
//  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& covariantVector = wrk_;
//  covariantVector.initialize();
//
//  fill_nodal_metrics(cell);
//
//  covariantVector.initialize();
//  curl.initialize();
//
//  for (std::size_t node=0; node < numNodes; ++node) {
//
//    const MeshScalarT j00 = jacobian(node, 0, 0);
//    const MeshScalarT j01 = jacobian(node, 0, 1);
//    const MeshScalarT j10 = jacobian(node, 1, 0);
//    const MeshScalarT j11 = jacobian(node, 1, 1);
//
//    covariantVector(node, 0 ) = j00*nodalVector(cell, node, level, 0) + j10*nodalVector(cell, node, level, 1);
//    covariantVector(node, 1 ) = j01*nodalVector(cell, node, level, 0) + j11*nodalVector(cell, node, level, 1);
//  }
//
//
//  for (std::size_t qp=0; qp < numQPs; ++qp) {
//    for (std::size_t node=0; node < numNodes; ++node) {
//
//      curl(qp) +=   covariantVector(node, 1)*grad_at_cub_points(node, qp,0)
//                  - covariantVector(node, 0)*grad_at_cub_points(node, qp,1);
//    }
//    curl(qp) = curl(qp)/jacobian_det(cell,qp);
//  }
//
//}

}
