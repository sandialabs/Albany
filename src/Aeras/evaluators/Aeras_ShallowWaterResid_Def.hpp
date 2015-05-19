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

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

#include "Shards_CellTopologyData.h"
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
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  mountainHeight  (p.get<std::string> ("Aeras Surface Height QP Variable Name"), dl->qp_scalar),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  source    (p.get<std::string> ("Shallow Water Source QP Variable Name"), dl->qp_vector),
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  spatialDim(p.get<std::size_t>("spatialDim")),
  //og not used
  //GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  lambda_nodal  (p.get<std::string>  ("Lambda Coord Nodal Name"), dl->node_scalar), 
  theta_nodal   (p.get<std::string>  ("Theta Coord Nodal Name"), dl->node_scalar), 
  gravity (Aeras::ShallowWaterConstants::self().gravity),
  hyperViscosity (p.get<std::string> ("Hyperviscosity Name"), dl->qp_vector),
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.))
{

  Teuchos::ParameterList* shallowWaterList = p.get<Teuchos::ParameterList*>("Shallow Water Problem");

  //IK, 3/25/14: boolean flag that says whether to integrate by parts the g*grad(h+hs) term
  // AGS: ToDo Add list validator!
  ibpGradH = shallowWaterList->get<bool>("IBP Grad h Term", false); //Default: false

  usePrescribedVelocity = shallowWaterList->get<bool>("Use Prescribed Velocity", false); //Default: false
  useHyperViscosity = shallowWaterList->get<bool>("Use Hyperviscosity", false); //Default: false
  
  
 #define ALBANY_VERBOSE
  
  AlphaAngle = shallowWaterList->get<double>("Rotation Angle", 0.0); //Default: 0.0

  const CellTopologyData *ctd = cellType->getCellTopologyData(); 
  int nNodes = ctd->node_count;
  int nDim   = ctd->dimension;
  std::string name     = ctd->name;
  size_t      len      = name.find("_");
  if (len != std::string::npos) name = name.substr(0,len);

#ifdef ALBANY_VERBOSE
  std::cout << "In Aeras::ShallowWaterResid: CellTopology is " << name << " with nodes " << nNodes 
            << "  dim " << nDim << std::endl;
  std::cout << "FullCellTopology name is " << ctd->name << std::endl;
#endif

  qpToNodeMap.resize(nNodes); 
  nodeToQPMap.resize(nNodes); 
  //Spectral quads
  if (name == "SpectralQuadrilateral" || name == "SpectralShellQuadrilateral") {
    if (nNodes == 4) {
       qpToNodeMap[0] = 0; qpToNodeMap[1] = 1; 
       qpToNodeMap[2] = 3; qpToNodeMap[3] = 2;  
       nodeToQPMap[0] = 0; nodeToQPMap[1] = 1; 
       nodeToQPMap[2] = 3; nodeToQPMap[3] = 2;  
    }
    else {
      for(int i=0; i<nNodes; i++) {
        qpToNodeMap[i] = i; 
        nodeToQPMap[i] = i; 
      }
    }
  }
  //Isoparametric quads
  else if (name == "Quadrilateral" || name == "ShellQuadrilateral") {
    if (nNodes == 4) {
      for(int i=0; i<nNodes; i++) {
        qpToNodeMap[i] = i; 
        nodeToQPMap[i] = i;
      } 
    }
    else if (nNodes == 9) {
      qpToNodeMap[0] = 0; qpToNodeMap[1] = 4; qpToNodeMap[2] = 1; 
      qpToNodeMap[3] = 7; qpToNodeMap[4] = 8; qpToNodeMap[5] = 5; 
      qpToNodeMap[6] = 3; qpToNodeMap[7] = 6; qpToNodeMap[8] = 2; 
      nodeToQPMap[0] = 0; nodeToQPMap[1] = 2; nodeToQPMap[2] = 8; 
      nodeToQPMap[3] = 6; nodeToQPMap[4] = 1; nodeToQPMap[5] = 5; 
      nodeToQPMap[6] = 7; nodeToQPMap[7] = 3; nodeToQPMap[8] = 4; 
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(true,
         Teuchos::Exceptions::InvalidParameter,"Aeras::ShallowWaterResid: qpToNodeMap and nodeToQPMap " << 
         "not implemented for quadrilateral/shellquadrilateral element with " << nNodes << ".");
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
       Teuchos::Exceptions::InvalidParameter,"Aeras::ShallowWaterResid: qpToNodeMap and nodeToQPMap " << 
       "non-quadrilateral/shellquadrilateral elements.");
  }
  
  this->addDependentField(U);
  this->addDependentField(UNodal);
  this->addDependentField(Ugrad);
  this->addDependentField(UDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  //this->addDependentField(GradBF);
  this->addDependentField(mountainHeight);
  this->addDependentField(sphere_coord);
  this->addDependentField(hyperViscosity);
  this->addDependentField(lambda_nodal);
  this->addDependentField(theta_nodal);
  this->addDependentField(source);

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
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  nodal_jacobian.resize(numNodes, 2, 2);
  nodal_inv_jacobian.resize(numNodes, 2, 2);
  nodal_det_j.resize(numNodes);
#endif
  cubature->getCubature(refPoints, refWeights);
  
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

  this->setName("Aeras::ShallowWaterResid"+PHX::typeAsString<EvalT>());


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
  this->registerSacadoParameter("Gravity", paramLib);
  this->registerSacadoParameter("Omega", paramLib);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //Allocationg additional data for Kokkos functors
  nodal_jacobian=Kokkos::View<MeshScalarT***, PHX::Device>("nodal_jacobian",numNodes,2,2);
  nodal_inv_jacobian=Kokkos::View<MeshScalarT***, PHX::Device>("nodal_inv_jacobian",numNodes,2,2); 
  nodal_det_j=Kokkos::View<MeshScalarT*, PHX::Device>("nodal_det_j",numNodes);
  refWeights_Kokkos=Kokkos::View<MeshScalarT*, PHX::Device>("refWeights_Kokkos",numQPs);
  grad_at_cub_points_Kokkos=Kokkos::View<MeshScalarT***, PHX::Device>("grad_at_cub_points_Kokkos",numNodes,numQPs,2);
  refPoints_kokkos=Kokkos::View<MeshScalarT**, PHX::Device>("refPoints_Kokkos",numQPs,2);

 for (int i=0; i<numQPs; i++)
 {
  refWeights_Kokkos(i)=refWeights(i);
  for (int j=0; j<2; j++){
   refPoints_kokkos(i,j)=refPoints(i,j);
    for (int k=0; k<numNodes; k++)
      grad_at_cub_points_Kokkos(k,i,j)=grad_at_cub_points(k,i,j);
  }
 } 

 std::vector<PHX::index_size_type> ddims_;
 ddims_.push_back(27);

 huAtNodes=PHX::MDField<ScalarT,Node,Dim>("huAtNodes",Teuchos::rcp(new PHX::MDALayout<Node,Dim>(numNodes,2)));
 huAtNodes.setFieldData(ViewFactory::buildView(huAtNodes.fieldTag(),ddims_));
 div_hU=PHX::MDField<ScalarT,QuadPoint>("div_hU",Teuchos::rcp(new PHX::MDALayout<QuadPoint>(numQPs)));
 div_hU.setFieldData(ViewFactory::buildView(div_hU.fieldTag(),ddims_));
 kineticEnergyAtNodes=PHX::MDField<ScalarT,Node>("kineticEnergyAtNodes",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 kineticEnergyAtNodes.setFieldData(ViewFactory::buildView(kineticEnergyAtNodes.fieldTag(),ddims_));
 gradKineticEnergy=PHX::MDField<ScalarT,QuadPoint,Dim>("gradKineticEnergy",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 gradKineticEnergy.setFieldData(ViewFactory::buildView(gradKineticEnergy.fieldTag(),ddims_));
 potentialEnergyAtNodes=PHX::MDField<ScalarT,Node>("potentialEnergyAtNodes",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 potentialEnergyAtNodes.setFieldData(ViewFactory::buildView(potentialEnergyAtNodes.fieldTag(),ddims_));
 gradPotentialEnergy=PHX::MDField<ScalarT,QuadPoint,Dim>("gradPotentialEnergy",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 gradPotentialEnergy.setFieldData(ViewFactory::buildView(gradPotentialEnergy.fieldTag(),ddims_));
 uAtNodes=PHX::MDField<ScalarT,Node,Dim>("uAtNodes",Teuchos::rcp(new PHX::MDALayout<Node,Dim>(numNodes,2)));
 uAtNodes.setFieldData(ViewFactory::buildView(uAtNodes.fieldTag(),ddims_));
 curlU=PHX::MDField<ScalarT,QuadPoint>("curlU",Teuchos::rcp(new PHX::MDALayout<QuadPoint>(numQPs)));
 curlU.setFieldData(ViewFactory::buildView(curlU.fieldTag(),ddims_));
 coriolis=PHX::MDField<ScalarT,QuadPoint>("coriolis",Teuchos::rcp(new PHX::MDALayout<QuadPoint>(numQPs)));
 coriolis.setFieldData(ViewFactory::buildView(coriolis.fieldTag(),ddims_));
 
 surf=PHX::MDField<ScalarT,Node>("surf",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 surf.setFieldData(ViewFactory::buildView(surf.fieldTag(),ddims_));
 surftilde=PHX::MDField<ScalarT,Node>("surftilde",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 surftilde.setFieldData(ViewFactory::buildView(surftilde.fieldTag(),ddims_));
 hgradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("hgradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 hgradNodes.setFieldData(ViewFactory::buildView(hgradNodes.fieldTag(),ddims_));
 htildegradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("htildegradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 htildegradNodes.setFieldData(ViewFactory::buildView(htildegradNodes.fieldTag(),ddims_));

 ucomp=PHX::MDField<ScalarT,Node>("ucomp",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 ucomp.setFieldData(ViewFactory::buildView(ucomp.fieldTag(),ddims_));
 vcomp=PHX::MDField<ScalarT,Node>("vcomp",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 vcomp.setFieldData(ViewFactory::buildView(vcomp.fieldTag(),ddims_));
 utildecomp=PHX::MDField<ScalarT,Node>("utildecomp",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 utildecomp.setFieldData(ViewFactory::buildView(utildecomp.fieldTag(),ddims_));
 vtildecomp=PHX::MDField<ScalarT,Node>("vtildecomp",Teuchos::rcp(new PHX::MDALayout<Node>(numNodes)));
 vtildecomp.setFieldData(ViewFactory::buildView(vtildecomp.fieldTag(),ddims_));

 ugradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("ugradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 ugradNodes.setFieldData(ViewFactory::buildView(ugradNodes.fieldTag(),ddims_));
 vgradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("vgradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 vgradNodes.setFieldData(ViewFactory::buildView(vgradNodes.fieldTag(),ddims_));
 utildegradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("utildegradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 utildegradNodes.setFieldData(ViewFactory::buildView(utildegradNodes.fieldTag(),ddims_));
 vtildegradNodes=PHX::MDField<ScalarT,QuadPoint,Dim>("vtildegradNodes",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 vtildegradNodes.setFieldData(ViewFactory::buildView(vtildegradNodes.fieldTag(),ddims_));
 utilde=PHX::MDField<ScalarT,QuadPoint,Dim>("utilde",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 utilde.setFieldData(ViewFactory::buildView(utilde.fieldTag(),ddims_));
 vtilde=PHX::MDField<ScalarT,QuadPoint,Dim>("vtilde",Teuchos::rcp(new PHX::MDALayout<QuadPoint,Dim>(numQPs,2)));
 vtilde.setFieldData(ViewFactory::buildView(vtilde.fieldTag(),ddims_));

 vcontra=PHX::MDField<ScalarT,Node,Dim>("vcontra",Teuchos::rcp(new PHX::MDALayout<Node,Dim>(numNodes,2)));
 vcontra.setFieldData(ViewFactory::buildView(vcontra.fieldTag(),ddims_));

nodeToQPMap_Kokkos=Kokkos::View<int*, PHX::Device> ("nodeToQPMap_Kokkos",nNodes);
for (int i=0; i<nNodes; i++)
 nodeToQPMap_Kokkos(i)=nodeToQPMap[i];
#endif
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
  //this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(mountainHeight,fm);

  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(hyperViscosity,fm);
  this->utils.setFieldData(lambda_nodal,fm);
  this->utils.setFieldData(theta_nodal,fm);
  this->utils.setFieldData(source,fm);

  this->utils.setFieldData(weighted_measure, fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_inv, fm);
  this->utils.setFieldData(jacobian_det, fm);

  this->utils.setFieldData(Residual,fm);
  
}

// *********************************************************************
//Kokkos functors
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT

template< typename ScalarT, typename ArrayT1,typename ArrayT2, typename ArrayJac, typename ArrayGrad>
KOKKOS_INLINE_FUNCTION
void gradient(const ArrayT1  & fieldAtNodes,
              const int &cell, ArrayT2  & gradField, ArrayJac &jacobian_inv, ArrayGrad &grad_at_cub_points_Kokkos) {


    for (int qp=0; qp < grad_at_cub_points_Kokkos.dimension(1); ++qp) {

      ScalarT gx = 0;
      ScalarT gy = 0;
      for (int node=0; node < grad_at_cub_points_Kokkos.dimension(0); ++node) {

       gx +=   fieldAtNodes(node)*grad_at_cub_points_Kokkos(node, qp,0);
       gy +=   fieldAtNodes(node)*grad_at_cub_points_Kokkos(node, qp,1);
      }

      gradField(qp, 0) = jacobian_inv(cell, qp, 0, 0)*gx + jacobian_inv(cell, qp, 1, 0)*gy;
      gradField(qp, 1) = jacobian_inv(cell, qp, 0, 1)*gx + jacobian_inv(cell, qp, 1, 1)*gy;
  }

}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_huAtNodes_vecDim3(const int& cell) const{
 for (int node=0; node < numNodes; ++node) {
      huAtNodes(node,0)= UNodal(cell,node,0)*UNodal(cell,node,1);
      huAtNodes(node,1)= UNodal(cell,node,0)*UNodal(cell,node,2);
 }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residual0(const int& cell) const{
 for (int node=0; node < numNodes; ++node) {
      surf(node) = UNodal(cell,node,0);
    }

    divergence(huAtNodes, cell);

    for (int qp=0; qp < numQPs; ++qp) {

      for (int node=0; node < numNodes; ++node) {

        Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell, node, qp)
                              +  div_hU(qp)*wBF(cell, node, qp); 
      }
    }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residual0_useHyperViscosity(const int& cell) const{
 for (int node=0; node < numNodes; ++node) {
      surf(node) = UNodal(cell,node,0);
    }

    divergence(huAtNodes, cell);

    for (int qp=0; qp < numQPs; ++qp) {

      for (int node=0; node < numNodes; ++node) {

        Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell, node, qp)
                              +  div_hU(qp)*wBF(cell, node, qp)
                              - hyperViscosity(cell,qp,0)*htildegradNodes(qp,0)*wGradBF(cell,node,qp,0)
                              - hyperViscosity(cell,qp,1)*htildegradNodes(qp,1)*wGradBF(cell,node,qp,1); 
      }
    }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residual3(const int& cell) const {

   for (std::size_t node=0; node < numNodes; ++node) 
      surf(node) = UNodal(cell,node,0);
    gradient<ScalarT>(surf, cell, hgradNodes, jacobian_inv, grad_at_cub_points_Kokkos);

   for (std::size_t qp=0; qp < numQPs; ++qp) {
     for (std::size_t node=0; node < numNodes; ++node) {
       Residual(cell,node,3) += U(cell,qp,3)*wBF(cell,node,qp) + hgradNodes(qp,0)*wGradBF(cell,node,qp,0)
                             + hgradNodes(qp,1)*wGradBF(cell,node,qp,1);
      }
   }

}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim3_usePrescribedVelocity_Tag& tag, const int& cell) const{
   for (int node=0; node < numNodes; ++node){
      Residual(cell,node,0)=0.0;
      Residual(cell,node,1)=0.0;
      Residual(cell,node,2)=0.0;
   }
   
 compute_huAtNodes_vecDim3(cell); 
 compute_Residual0(cell);

  for (int qp=0; qp < numQPs; ++qp) {
        for (int node=0; node < numNodes; ++node) {
          Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp) + source(cell,qp,1)*wBF(cell, node, qp);
          Residual(cell,node,2) += UDot(cell,qp,2)*wBF(cell,node,qp) + source(cell,qp,2)*wBF(cell, node, qp);
        }
      }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim4_Tag& tag, const int& cell) const{
   for (int node=0; node < numNodes; ++node){
      Residual(cell,node,0)=0.0;
      Residual(cell,node,1)=0.0;
      Residual(cell,node,2)=0.0;
   }
   
 compute_huAtNodes_vecDim3(cell); 
 compute_Residual0_useHyperViscosity(cell);
 compute_Residual3(cell);

  //FIXME!
  for (int qp=0; qp < numQPs; ++qp) {
        for (int node=0; node < numNodes; ++node) {
          Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp) + source(cell,qp,1)*wBF(cell, node, qp);
          Residual(cell,node,2) += UDot(cell,qp,2)*wBF(cell,node,qp) + source(cell,qp,2)*wBF(cell, node, qp);
        }
   }
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Tag& tag, const int& cell) const{

 for (int node=0; node < numNodes; ++node){
      Residual(cell,node,0)=0.0;
      Residual(cell,node,1)=0.0;
      Residual(cell,node,2)=0.0;
   }

 compute_huAtNodes_vecDim3(cell);
 compute_Residual0(cell);

 get_coriolis(cell);

      for (int node=0; node < numNodes; ++node) {
        ScalarT depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]);
        ScalarT ulambda = UNodal(cell, node,1);
        ScalarT utheta  = UNodal(cell, node,2);
        kineticEnergyAtNodes(node) = 0.5*(ulambda*ulambda + utheta*utheta);
        uAtNodes(node, 0) = ulambda;
        uAtNodes(node, 1) = utheta;
        ucomp(node) = ulambda;
        vcomp(node) = utheta;
      }
     gradient<ScalarT>(kineticEnergyAtNodes, cell, gradKineticEnergy, jacobian_inv, grad_at_cub_points_Kokkos);
      curl(cell);

     for (int qp=0; qp < numQPs; ++qp) {
          for (int node=0; node < numNodes; ++node) {
            Residual(cell,node,1) += ( UDot(cell,qp,1) + gradKineticEnergy(qp,0) - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,0) + source(cell,qp,1)*wBF(cell,node,qp);
            Residual(cell,node,2) += ( UDot(cell,qp,2) + gradKineticEnergy(qp,1) + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,1) + source(cell,qp,2)*wBF(cell,node,qp);
          }
        }

}
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim6_Tag& tag, const int& cell) const{

 for (int node=0; node < numNodes; ++node){
      Residual(cell,node,0)=0.0;
      Residual(cell,node,1)=0.0;
      Residual(cell,node,2)=0.0;
   }

 compute_huAtNodes_vecDim3(cell);
 compute_Residual0_useHyperViscosity(cell);
 compute_Residual3(cell);

 get_coriolis(cell);

      for (int node=0; node < numNodes; ++node) {
        ScalarT depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]);
        ScalarT ulambda = UNodal(cell, node,1);
        ScalarT utheta  = UNodal(cell, node,2);
        kineticEnergyAtNodes(node) = 0.5*(ulambda*ulambda + utheta*utheta);
        uAtNodes(node, 0) = ulambda;
        uAtNodes(node, 1) = utheta;
        ucomp(node) = ulambda;
        vcomp(node) = utheta;
        utildecomp(node) = UNodal(cell,node,4);
        vtildecomp(node) = UNodal(cell,node,5);
      }

      //obtain grads of U, V comp
      gradient<ScalarT>(utildecomp, cell, utilde, jacobian_inv, grad_at_cub_points_Kokkos);
      gradient<ScalarT>(vtildecomp, cell, vtildegradNodes, jacobian_inv, grad_at_cub_points_Kokkos);
      
      gradient<ScalarT>(kineticEnergyAtNodes, cell, gradKineticEnergy, jacobian_inv, grad_at_cub_points_Kokkos);
      curl(cell);

     for (int qp=0; qp < numQPs; ++qp) {
          for (int node=0; node < numNodes; ++node) {
            Residual(cell,node,1) += ( UDot(cell,qp,1) + gradKineticEnergy(qp,0) 
                                  - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,0) + source(cell,qp,1)*wBF(cell,node,qp);
                                  -  hyperViscosity(cell,qp,0)*utildegradNodes(qp,0)*wGradBF(cell,node,qp,0) 
                                  -  hyperViscosity(cell,qp,1)*utildegradNodes(qp,1)*wGradBF(cell,node,qp,1);   
            Residual(cell,node,2) += ( UDot(cell,qp,2) + gradKineticEnergy(qp,1) 
                                  + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1))*wBF(cell,node,qp)
                                  - gravity*U(cell,qp,0)*wGradBF(cell,node,qp,1) + source(cell,qp,2)*wBF(cell,node,qp);
                                  - hyperViscosity(cell,qp,0)*vtildegradNodes(qp,0)*wGradBF(cell,node,qp,0) 
                                  -  hyperViscosity(cell,qp,1)*vtildegradNodes(qp,1)*wGradBF(cell,node,qp,1);   
            Residual(cell,node,4) += U(cell,qp,4)*wBF(cell,node,qp) + ugradNodes(qp,0)*wGradBF(cell,node,qp,0)
                                  + ugradNodes(qp,1)*wGradBF(cell,node,qp,1);
            Residual(cell,node,5) += U(cell,qp,5)*wBF(cell,node,qp) + vgradNodes(qp,0)*wGradBF(cell,node,qp,0)
                                  + vgradNodes(qp,1)*wGradBF(cell,node,qp,1);
          }
        }

}

template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::divergence(const PHX::MDField<ScalarT,Node, Dim>  & fieldAtNodes,
      const int cell) const  {


  fill_nodal_metrics(cell);

  for (std::size_t node=0; node < numNodes; ++node) {

    const MeshScalarT jinv00 = nodal_inv_jacobian(node, 0, 0);
    const MeshScalarT jinv01 = nodal_inv_jacobian(node, 0, 1);
    const MeshScalarT jinv10 = nodal_inv_jacobian(node, 1, 0);
    const MeshScalarT jinv11 = nodal_inv_jacobian(node, 1, 1);

    vcontra(node, 0 ) = nodal_det_j(node)*(
        jinv00*fieldAtNodes(node, 0) + jinv01*fieldAtNodes(node, 1) );
    vcontra(node, 1 ) = nodal_det_j(node)*(
        jinv10*fieldAtNodes(node, 0)+ jinv11*fieldAtNodes(node, 1) );
  }


  for (int qp=0; qp < numQPs; ++qp) {
    for (int node=0; node < numNodes; ++node) {
//      ScalarT tempAdd =vcontra(node, 0)*grad_at_cub_points_Kokkos(node, qp,0)
//                  + vcontra(node, 1)*grad_at_cub_points_Kokkos(node, qp,1);
 //     Kokkos::atomic_fetch_add(&div_hU(qp), tempAdd);
      div_hU(qp) +=   vcontra(node, 0)*grad_at_cub_points_Kokkos(node, qp,0)
                  + vcontra(node, 1)*grad_at_cub_points_Kokkos(node, qp,1);
    }

  }

  for (int qp=0; qp < numQPs; ++qp) {
    div_hU(qp) = div_hU(qp)/jacobian_det(cell,qp);
  }

}

template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::fill_nodal_metrics(const int &cell) const {

  for (size_t v = 0; v < numNodes; ++v) {
    int qp = nodeToQPMap_Kokkos[v];

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
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::curl(const int &cell) const {


  fill_nodal_metrics(cell);


  for (int node=0; node < numNodes; ++node) {

    const MeshScalarT j00 = nodal_jacobian(node, 0, 0);
    const MeshScalarT j01 = nodal_jacobian(node, 0, 1);
    const MeshScalarT j10 = nodal_jacobian(node, 1, 0);
    const MeshScalarT j11 = nodal_jacobian(node, 1, 1);

    vcontra(node, 0 ) = j00*uAtNodes(node, 0) + j10*uAtNodes(node, 1);
    vcontra(node, 1 ) = j01*uAtNodes(node, 0) + j11*uAtNodes(node, 1);
  }


  for (int qp=0; qp < numQPs; ++qp) {
    for (int node=0; node < numNodes; ++node) {

      curlU(qp) +=   vcontra(node, 1)*grad_at_cub_points_Kokkos(node, qp,0)
                  - vcontra(node, 0)*grad_at_cub_points_Kokkos(node, qp,1);
    }
    curlU(qp) = curlU(qp)/jacobian_det(cell,qp);
  }


}

template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::get_coriolis(const int &cell) const {

  double alpha = AlphaAngle; /*must match what is in initial condition for TC2 and TC5.
                      //see AAdatpt::AerasZonal analytic function. */
                      //
  for (int qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
  }
}
#endif
//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  PHAL::set(Residual, 0.0);

//Note that vars huAtNodes, div_hU, ... below are redefined locally here. 
//Global vars with such names exist too (see constructor).
  Intrepid::FieldContainer<ScalarT>  huAtNodes(numNodes,2);
  Intrepid::FieldContainer<ScalarT>  div_hU(numQPs);
  Intrepid::FieldContainer<ScalarT>  kineticEnergyAtNodes(numNodes);
  Intrepid::FieldContainer<ScalarT>  gradKineticEnergy(numQPs,2);
  Intrepid::FieldContainer<ScalarT>  potentialEnergyAtNodes(numNodes);
  Intrepid::FieldContainer<ScalarT>  gradPotentialEnergy(numQPs,2);
  Intrepid::FieldContainer<ScalarT>  uAtNodes(numNodes, 2);
  Intrepid::FieldContainer<ScalarT>  curlU(numQPs);
  Intrepid::FieldContainer<ScalarT>  coriolis(numQPs);

  //container for surface height for viscosty
  Intrepid::FieldContainer<ScalarT> surf(numNodes);
  Intrepid::FieldContainer<ScalarT> surftilde(numNodes);
  //conteiner for surface height gradient for viscosity
  Intrepid::FieldContainer<ScalarT> hgradNodes(numQPs,2);
  Intrepid::FieldContainer<ScalarT> htildegradNodes(numQPs,2);
  
  //containers for U and V components separately, I don't know how to
  //pass to the gradient uAtNodes(:,0)
  Intrepid::FieldContainer<ScalarT> ucomp(numNodes);
  Intrepid::FieldContainer<ScalarT> vcomp(numNodes);
  Intrepid::FieldContainer<ScalarT> utildecomp(numNodes);
  Intrepid::FieldContainer<ScalarT> vtildecomp(numNodes);
  //containers for grads of velocity U, V components for viscosity
  //note that we do not implement it for the most generality (any dimension velocity)
  //because the rest of the code considers only 2D velocity (look at definition of uAtNodes)
  Intrepid::FieldContainer<ScalarT> ugradNodes(numQPs,2);
  Intrepid::FieldContainer<ScalarT> vgradNodes(numQPs,2);
  Intrepid::FieldContainer<ScalarT> utildegradNodes(numQPs,2);
  Intrepid::FieldContainer<ScalarT> vtildegradNodes(numQPs,2);
 
//TODO: erase presence of ibpGradH
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      
    // Depth Equation (Eq# 0)
    huAtNodes.initialize();
    div_hU.initialize();
    
    surf.initialize();
    surftilde.initialize();
    hgradNodes.initialize();
    htildegradNodes.initialize();

    for (std::size_t node=0; node < numNodes; ++node) {
      ScalarT surfaceHeight = UNodal(cell,node,0);
      ScalarT ulambda = UNodal(cell, node,1);
      ScalarT utheta  = UNodal(cell, node,2);
      huAtNodes(node,0) = surfaceHeight*ulambda;
      huAtNodes(node,1) = surfaceHeight*utheta;
    }
    
    for (std::size_t node=0; node < numNodes; ++node) 
      surf(node) = UNodal(cell,node,0);
    gradient(surf, cell, hgradNodes);

    if (useHyperViscosity) {
      for (std::size_t node=0; node < numNodes; ++node) 
        surftilde(node) = UNodal(cell,node,3);
      gradient(surftilde, cell, htildegradNodes);
    }

    divergence(huAtNodes, cell, div_hU);

    for (std::size_t qp=0; qp < numQPs; ++qp) {
        
      for (std::size_t node=0; node < numNodes; ++node) {

        Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell, node, qp)
                              +  div_hU(qp)*wBF(cell, node, qp); 
        if (useHyperViscosity) { //hyperviscosity residual(0) = residual(0) - tau*grad(htilde)*grad(phi) 
          Residual(cell,node,0) -= hyperViscosity(cell,qp,0)*htildegradNodes(qp,0)*wGradBF(cell,node,qp,0) 
                                -  hyperViscosity(cell,qp,1)*htildegradNodes(qp,1)*wGradBF(cell,node,qp,1);   
        }
      }
    }
    
  }

  if (useHyperViscosity) { //hyperviscosity residual(3) = htilde*phi + grad(h)*grad(phi) 

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      surf.initialize();
      hgradNodes.initialize();
      for (std::size_t node=0; node < numNodes; ++node) 
        surf(node) = UNodal(cell,node,0);
      gradient(surf, cell, hgradNodes); 

      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t node=0; node < numNodes; ++node) {
            Residual(cell,node,3) += U(cell,qp,3)*wBF(cell,node,qp) + hgradNodes(qp,0)*wGradBF(cell,node,qp,0)
                                  + hgradNodes(qp,1)*wGradBF(cell,node,qp,1);
        }
      }
    }
  }// endif use hyperviscosity
  // Velocity Equations
  if (usePrescribedVelocity) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t node=0; node < numNodes; ++node) {
          Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp) + source(cell,qp,1)*wBF(cell, node, qp);
          Residual(cell,node,2) += UDot(cell,qp,2)*wBF(cell,node,qp) + source(cell,qp,2)*wBF(cell, node, qp); 
        }
      }
    }
  }
  else { // Solve for velocity

    // Velocity Equations (Eq# 1,2)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {

      potentialEnergyAtNodes.initialize();
      gradPotentialEnergy.initialize();

      kineticEnergyAtNodes.initialize();
      gradKineticEnergy.initialize();
      uAtNodes.initialize();
      
      ucomp.initialize();
      vcomp.initialize();
      ugradNodes.initialize();
      vgradNodes.initialize();
      utildecomp.initialize();
      vtildecomp.initialize();
      utildegradNodes.initialize();
      vtildegradNodes.initialize();

      get_coriolis(cell, coriolis);

      for (std::size_t node=0; node < numNodes; ++node) {
        ScalarT depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap[node]);
        ScalarT ulambda = UNodal(cell, node,1);
        ScalarT utheta  = UNodal(cell, node,2);
        kineticEnergyAtNodes(node) = 0.5*(ulambda*ulambda + utheta*utheta);

        potentialEnergyAtNodes(node) = gravity*depth;

        uAtNodes(node, 0) = ulambda;
        uAtNodes(node, 1) = utheta;
        
        //for viscosity
        ucomp(node) = ulambda;
        vcomp(node) = utheta;
        if (useHyperViscosity) {
           utildecomp(node) = UNodal(cell,node,4);
           vtildecomp(node) = UNodal(cell,node,5);
        } 
      }
      
      //obtain grads of U, V comp
      gradient(ucomp, cell, ugradNodes);
      gradient(vcomp, cell, vgradNodes);
      if (useHyperViscosity) {
        gradient(utildecomp, cell, utildegradNodes);
        gradient(vtildecomp, cell, vtildegradNodes);
      }

      gradient(potentialEnergyAtNodes, cell, gradPotentialEnergy);

      gradient(kineticEnergyAtNodes, cell, gradKineticEnergy);
      curl(uAtNodes, cell, curlU);
 
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t node=0; node < numNodes; ++node) {
        
          Residual(cell,node,1) += (   UDot(cell,qp,1) + gradKineticEnergy(qp,0)
                                       + gradPotentialEnergy(qp,0)
                                       - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2)
                                      )*wBF(cell,node,qp); 
          Residual(cell,node,2) += (   UDot(cell,qp,2) + gradKineticEnergy(qp,1)
                                       + gradPotentialEnergy(qp,1)
                                       + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1)
                                      )*wBF(cell,node,qp); 
          if (useHyperViscosity) {
                                    //hyperviscosity residual(1) = residual(1) - tau*grad(utilde)*grad(phi) 
            Residual(cell,node,1) -= hyperViscosity(cell,qp,0)*utildegradNodes(qp,0)*wGradBF(cell,node,qp,0) 
                                  -  hyperViscosity(cell,qp,1)*utildegradNodes(qp,1)*wGradBF(cell,node,qp,1);   
                                    //hyperviscosity residual(2) = residual(2) - tau*grad(vtilde)*grad(phi) 
            Residual(cell,node,2) -= hyperViscosity(cell,qp,0)*vtildegradNodes(qp,0)*wGradBF(cell,node,qp,0) 
                                  -  hyperViscosity(cell,qp,1)*vtildegradNodes(qp,1)*wGradBF(cell,node,qp,1);   
                                    //hyperviscosity residual(4) = utilde*phi + grad(u)*grad(phi) 
            Residual(cell,node,4) += U(cell,qp,4)*wBF(cell,node,qp) + ugradNodes(qp,0)*wGradBF(cell,node,qp,0)
                                  + ugradNodes(qp,1)*wGradBF(cell,node,qp,1);
                                  //hyperviscosity residual(5) = vtilde*phi + grad(v)*grap(phi)
            Residual(cell,node,5) += U(cell,qp,5)*wBF(cell,node,qp) + vgradNodes(qp,0)*wGradBF(cell,node,qp,0)
                                  + vgradNodes(qp,1)*wGradBF(cell,node,qp,1);
          }
        }
      }
    } // end cell loop
  } //end if !prescribedVelocities
#else
a = Aeras::ShallowWaterConstants::self().earthRadius;
myPi = Aeras::ShallowWaterConstants::self().pi;

  if (usePrescribedVelocity) {
      if (useHyperViscosity)
        Kokkos::parallel_for(ShallowWaterResid_VecDim4_Policy(0,workset.numCells),*this); 
      else
        Kokkos::parallel_for(ShallowWaterResid_VecDim3_usePrescribedVelocity_Policy(0,workset.numCells),*this); 
  }
  else {
     if (useHyperViscosity)
       Kokkos::parallel_for(ShallowWaterResid_VecDim6_Policy(0,workset.numCells),*this);
     else
       Kokkos::parallel_for(ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Policy(0,workset.numCells),*this);
  }

#endif
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
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
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
        jinv10*fieldAtNodes(node, 0)+ jinv11*fieldAtNodes(node, 1) );
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
  double alpha = AlphaAngle; /*must match what is in initial condition for TC2 and TC5.
                      //see AAdatpt::AerasZonal analytic function. */

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
  }

}
#endif  
}
