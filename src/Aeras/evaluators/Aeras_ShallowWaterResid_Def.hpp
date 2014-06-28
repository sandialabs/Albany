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

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_ShallowWaterConstants.hpp"
namespace Aeras {


//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterResid<EvalT, Traits>::
ShallowWaterResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  U        (p.get<std::string> ("QP Variable Name"), dl->qp_vector),
  UNodal   (p.get<std::string> ("Nodal Variable Name"), dl->node_vector),
  Ugrad    (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  UDot     (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_vector),
  mountainHeight  (p.get<std::string> ("Aeras Surface Height QP Variable Name"), dl->qp_scalar),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  spatialDim(p.get<std::size_t>("spatialDim")),
  GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  gravity (Aeras::ShallowWaterConstants::self().gravity),
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.))
{

  Teuchos::ParameterList* shallowWaterList = p.get<Teuchos::ParameterList*>("Shallow Water Problem");

  //IK, 3/25/14: boolean flag that says whether to integrate by parts the g*grad(h+hs) term
  // AGS: ToDo Add list validator!
  ibpGradH = shallowWaterList->get<bool>("IBP Grad h Term", false); //Default: false

  usePrescribedVelocity = shallowWaterList->get<bool>("Use Prescribed Velocity", false); //Default: false

  this->addDependentField(U);
  this->addDependentField(UNodal);
  this->addDependentField(Ugrad);
  this->addDependentField(UDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(GradBF);
  this->addDependentField(mountainHeight);
  this->addDependentField(sphere_coord);

  this->addDependentField(weighted_measure);
  this->addDependentField(jacobian);
  this->addDependentField(jacobian_inv);
  this->addDependentField(jacobian_det);

  this->addEvaluatedField(Residual);

  std::vector<PHX::DataLayout::size_type> dims;
    wGradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

  refWeights        .resize               (numQPs);
  grad_at_cub_points.resize     (numNodes, numQPs, 2);
  refPoints         .resize               (numQPs, 2);
  nodal_jacobian.resize(numNodes, 2, 2);
  nodal_inv_jacobian.resize(numNodes, 2, 2);
  nodal_det_j.resize(numNodes);

  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

   this->setName("Aeras::ShallowWaterResid"+PHX::TypeString<EvalT>::value);


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  std::vector<PHX::DataLayout::size_type> gradDims;
  wGradBF.fieldTag().dataLayout().dimensions(gradDims);


  gradDims.clear();
  Ugrad.fieldTag().dataLayout().dimensions(gradDims);


//  std::cout << " vecDim = " << vecDim << std::endl;
//  std::cout << " numDims = " << numDims << std::endl;
//  std::cout << " numQPs = " << numQPs << std::endl;
//  std::cout << " numNodes = " << numNodes << std::endl;

  // Register Reynolds number as Sacado-ized Parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Gravity", this, paramLib);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Omega", this, paramLib);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UNodal,fm);
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(mountainHeight,fm);

  this->utils.setFieldData(sphere_coord,fm);

  this->utils.setFieldData(weighted_measure, fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_inv, fm);
  this->utils.setFieldData(jacobian_det, fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  for (std::size_t i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  Intrepid::FieldContainer<ScalarT>  huAtNodes(numNodes,2);
  Intrepid::FieldContainer<ScalarT>  div_hU(numQPs);
  Intrepid::FieldContainer<ScalarT>  kineticEnergyAtNodes(numNodes);
  Intrepid::FieldContainer<ScalarT>  gradKineticEnergy(numQPs,2);
  Intrepid::FieldContainer<ScalarT>  potentialEnergyAtNodes(numNodes);
  Intrepid::FieldContainer<ScalarT>  gradPotentialEnergy(numQPs,2);
  Intrepid::FieldContainer<ScalarT>  uAtNodes(numNodes, 2);
  Intrepid::FieldContainer<ScalarT>  curlU(numQPs);
  Intrepid::FieldContainer<ScalarT>  coriolis(numQPs);


  for (std::size_t cell=0; cell < workset.numCells; ++cell) {


    // Depth Equation (Eq# 0)
    huAtNodes.initialize();
    div_hU.initialize();

    for (std::size_t node=0; node < numNodes; ++node) {
      ScalarT surfaceHeight = UNodal(cell,node,0);
      ScalarT ulambda = UNodal(cell, node,1);
      ScalarT utheta  = UNodal(cell, node,2);
      huAtNodes(node,0) = surfaceHeight*ulambda;
      huAtNodes(node,1) = surfaceHeight*utheta;
    }

    divergence(huAtNodes, cell, div_hU);


    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t node=0; node < numNodes; ++node) {

        Residual(cell,node,0) +=  UDot(cell,qp,0)*wBF(cell, node, qp) +  div_hU(qp)*wBF(cell, node, qp);
      }
    }


  }

  // Velocity Equations
  if (usePrescribedVelocity) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t node=0; node < numNodes; ++node) {
          Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp);
          Residual(cell,node,2) += UDot(cell,qp,2)*wBF(cell,node,qp);
        }
      }
    }
  }
  else { // Solve for velocity

    // Velocity Equations (Eq# 1,2)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {


      if (ibpGradH == false) {  //do not integrate by parts the grad h term 
        potentialEnergyAtNodes.initialize();
        gradPotentialEnergy.initialize();
      }
      kineticEnergyAtNodes.initialize();
      gradKineticEnergy.initialize();
      uAtNodes.initialize();

      get_coriolis(cell, coriolis);

      for (std::size_t node=0; node < numNodes; ++node) {
        ScalarT depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap[node]);
        ScalarT ulambda = UNodal(cell, node,1);
        ScalarT utheta  = UNodal(cell, node,2);
        kineticEnergyAtNodes(node) = 0.5*(ulambda*ulambda + utheta*utheta);
        if (ibpGradH == false)  //do not integrate by parts the grad h term 
          potentialEnergyAtNodes(node) = gravity*depth;
        uAtNodes(node, 0) = ulambda;
        uAtNodes(node, 1) = utheta;

      }
      if (ibpGradH == false) 
        gradient(potentialEnergyAtNodes, cell, gradPotentialEnergy);

      gradient(kineticEnergyAtNodes, cell, gradKineticEnergy);
      curl(uAtNodes, cell, curlU);
 
      if (ibpGradH == false) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t node=0; node < numNodes; ++node) {
            Residual(cell,node,1) += ( UDot(cell,qp,1) + gradKineticEnergy(qp,0) + gradPotentialEnergy(qp,0) - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2)
            )*wBF(cell,node,qp);
            Residual(cell,node,2) += ( UDot(cell,qp,2) + gradKineticEnergy(qp,1) + gradPotentialEnergy(qp,1) + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1)
            )*wBF(cell,node,qp);
          }
        }
      }
      else { //integrate by parts the grad h term
        //is transformation required to define divergence on wGradBF??   Need to figure this out (IK, 3/30/14).  Code below does not work yet as is.  
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t node=0; node < numNodes; ++node) {
            Residual(cell,node,1) += ( UDot(cell,qp,1) + gradKineticEnergy(qp,0) - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,0) ;
            Residual(cell,node,2) += ( UDot(cell,qp,2) + gradKineticEnergy(qp,1) + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,1) ;
          }
        }
      }
    }
  }
}

//**********************************************************************
// Provide Access to Parameter for sensitivity/optimization/UQ
template<typename EvalT,typename Traits>
typename ShallowWaterResid<EvalT,Traits>::ScalarT&
ShallowWaterResid<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="Gravity") return gravity;
  else if (n=="Omega") return Omega;
}
//**********************************************************************

template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::divergence(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
    std::size_t cell, Intrepid::FieldContainer<ScalarT>  & div) {

  Intrepid::FieldContainer<ScalarT>  vcontra(numNodes, 2);

  fill_nodal_metrics(cell);

  vcontra.initialize();
  div.initialize();

  for (std::size_t node=0; node < numNodes; ++node) {

    const MeshScalarT jinv00 = nodal_inv_jacobian(node, 0, 0);
    const MeshScalarT jinv01 = nodal_inv_jacobian(node, 0, 1);
    const MeshScalarT jinv10 = nodal_inv_jacobian(node, 1, 0);
    const MeshScalarT jinv11 = nodal_inv_jacobian(node, 1, 1);

    vcontra(node, 0 ) = nodal_det_j(node)*(
        jinv00*fieldAtNodes(node, 0) + jinv01*fieldAtNodes(node, 1) );
    vcontra(node, 1 ) = nodal_det_j(node)*(
        jinv10*fieldAtNodes(node, 0) + jinv11*fieldAtNodes(node, 1) );
  }


  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t node=0; node < numNodes; ++node) {

      div(qp) +=   vcontra(node, 0)*grad_at_cub_points(node, qp,0)
                  + vcontra(node, 1)*grad_at_cub_points(node, qp,1);
    }

  }

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    div(qp) = div(qp)/jacobian_det(cell,qp);
  }

//  for(size_t v = 0; v < numNodes; ++v) {
//    for(size_t q = 0; q < numQPs; ++q) {
//      div(q) += jacobian_inv(cell,q,0,0)*grad_at_cub_points(v, q, 0)*fieldAtNodes(v, 0) +
//          jacobian_inv(cell,q,0,1)*grad_at_cub_points(v, q, 0)*fieldAtNodes(v, 1) +
//          jacobian_inv(cell,q,1,0)*grad_at_cub_points(v, q, 1)*fieldAtNodes(v, 0) +
//          jacobian_inv(cell,q,1,1)*grad_at_cub_points(v, q, 1)*fieldAtNodes(v, 1);
//
//    }
//  }
}
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::gradient(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
    std::size_t cell, Intrepid::FieldContainer<ScalarT>  & gradField) {

  gradField.initialize();

    for (std::size_t qp=0; qp < numQPs; ++qp) {

      ScalarT gx = 0;
      ScalarT gy = 0;
      for (std::size_t node=0; node < numNodes; ++node) {

       gx +=   fieldAtNodes(node)*grad_at_cub_points(node, qp,0);
       gy +=   fieldAtNodes(node)*grad_at_cub_points(node, qp,1);
      }

      gradField(qp, 0) = jacobian_inv(cell, qp, 0, 0)*gx + jacobian_inv(cell, qp, 1, 0)*gy;
      gradField(qp, 1) = jacobian_inv(cell, qp, 0, 1)*gx + jacobian_inv(cell, qp, 1, 1)*gy;
  }

}
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::fill_nodal_metrics(std::size_t cell) {

  nodal_jacobian.initialize();
  nodal_det_j.initialize();
  nodal_inv_jacobian.initialize();

  for (size_t v = 0; v < numNodes; ++v) {
    int qp = nodeToQPMap[v];

    for (size_t b1 = 0; b1 < 2; ++b1) {
      for (size_t b2 = 0; b2 < 2; ++b2) {

        nodal_jacobian(v, b1, b2) = jacobian(cell, qp,b1, b2);
        nodal_inv_jacobian(v, b1, b2) = jacobian_inv(cell, qp,b1, b2);
      }
    }
    nodal_det_j(v) = jacobian_det(cell, qp);
  }
  return;

}

template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::curl(const Intrepid::FieldContainer<ScalarT>  & nodalVector,
    std::size_t cell, Intrepid::FieldContainer<ScalarT>  & curl) {

  Intrepid::FieldContainer<ScalarT>  covariantVector(numNodes, 2);

  fill_nodal_metrics(cell);

  covariantVector.initialize();
  curl.initialize();

  for (std::size_t node=0; node < numNodes; ++node) {

    const MeshScalarT j00 = nodal_jacobian(node, 0, 0);
    const MeshScalarT j01 = nodal_jacobian(node, 0, 1);
    const MeshScalarT j10 = nodal_jacobian(node, 1, 0);
    const MeshScalarT j11 = nodal_jacobian(node, 1, 1);

    covariantVector(node, 0 ) = j00*nodalVector(node, 0) + j10*nodalVector(node, 1);
    covariantVector(node, 1 ) = j01*nodalVector(node, 0) + j11*nodalVector(node, 1);
  }


  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t node=0; node < numNodes; ++node) {

      curl(qp) +=   covariantVector(node, 1)*grad_at_cub_points(node, qp,0)
                  - covariantVector(node, 0)*grad_at_cub_points(node, qp,1);
    }
    curl(qp) = curl(qp)/jacobian_det(cell,qp);
  }


}

template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis) {

  coriolis.initialize();
  double alpha = 0;  /*must match what is in initial condition for TC2 and TC5.
                      see AAdatpt::AerasZonal analytic function. */

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
  }

}
}
