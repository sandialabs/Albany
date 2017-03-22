//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
#include "Aeras_ShallowWaterConstants.hpp"

#include "Shards_CellTopologyData.h"
namespace Aeras {


//OG: A debugging statement prints evaluator's name
//#define AERAS_OUTPUT
//#define ALBANY_KOKKOS_UNDER_DEVELOPMENT


//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterResid<EvalT, Traits>::
ShallowWaterResid(const Teuchos::ParameterList& p,
  const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  GradBF  (p.get<std::string> ("Gradient BF Name"),dl->node_qp_gradient),
  U        (p.get<std::string> ("QP Variable Name"), dl->node_vector),
  UNodal   (p.get<std::string> ("Nodal Variable Name"), dl->node_vector),
  UDot     (p.get<std::string> ("QP Time Derivative Variable Name"), dl->node_vector),
  //OG UDotDot is not in use?
  /*
  UDotDot     (p.get<std::string> ("Time Dependent Variable Name"), dl->node_vector),
  UDotDotNodal   (p.get<std::string> ("Time Dependent Nodal Variable Name"), dl->node_vector),
  */
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  mountainHeight  (p.get<std::string> ("Aeras Surface Height QP Variable Name"), dl->qp_scalar),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  /*
  source    (p.get<std::string> ("Shallow Water Source QP Variable Name"), dl->qp_vector),
  */
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  spatialDim(p.get<std::size_t>("spatialDim")),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  lambda_nodal  (p.get<std::string>  ("Lambda Coord Nodal Name"), dl->node_scalar),
  theta_nodal   (p.get<std::string>  ("Theta Coord Nodal Name"), dl->node_scalar),
  gravity (Aeras::ShallowWaterConstants::self().gravity),
  /*
  hyperviscosity (p.get<std::string> ("Hyperviscosity Name"), dl->qp_vector),
  */
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.)),
  RRadius(1.0/Aeras::ShallowWaterConstants::self().earthRadius),
  doNotDampRotation(true)
{

#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::Constructor" << std::endl;
#endif

  //OG I noticed that source(...,0) is never used, it means TC4 is broken.

  Teuchos::ParameterList* shallowWaterList = p.get<Teuchos::ParameterList*>("Shallow Water Problem");

  usePrescribedVelocity = shallowWaterList->get<bool>("Use Prescribed Velocity", false); //Default: false

  useImplHyperviscosity = shallowWaterList->get<bool>("Use Implicit Hyperviscosity", false); //Default: false

  useExplHyperviscosity = shallowWaterList->get<bool>("Use Explicit Hyperviscosity", false); //Default: false

  plotVorticity = shallowWaterList->get<bool>("Plot Vorticity", false); //Default: false

  //OG: temporary, because if tau=0, sqrt(hyperviscosity(:)) leads to nans in laplace.
  //maybe, changing order of operation, like multiplying by tau later would help?
  sHvTau = sqrt(shallowWaterList->get<double>("Hyperviscosity Tau", 0.0));

  if( useImplHyperviscosity && useExplHyperviscosity )
    TEUCHOS_TEST_FOR_EXCEPTION(true,
 		               Teuchos::Exceptions::InvalidParameter,"Aeras::ShallowWaterResid: " <<
			       "The namelist sets useImplHyperviscosity = true and useExplHyperviscosity = true. " <<
			       "Using both explicit and implicit hyperviscosity is not possible. "<<
			       "Set useImplHyperviscosity or useImplHyperviscosity (or both) to false.");

  //#define ALBANY_VERBOSE

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
    TEUCHOS_TEST_FOR_EXCEPTION(true,
                               Teuchos::Exceptions::InvalidParameter,"Aeras::ShallowWaterResid no longer works with isoparameteric " <<
			       "Quads/ShellQuads! Please re-run with spectral elements (IKT, 7/30/2015).");
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
  this->addDependentField(UDot);
  //OG UDotDot is not in use?
  /*
  this->addDependentField(UDotDot);
  this->addDependentField(UDotDotNodal);
  */
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(GradBF);
  this->addDependentField(mountainHeight);
  this->addDependentField(sphere_coord);
  /*
  this->addDependentField(hyperviscosity);
  */
  this->addDependentField(lambda_nodal);
  this->addDependentField(theta_nodal);
  /*
  this->addDependentField(source);
  */
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
  numCells = dims[0];

  if (nNodes != numQPs) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
 			       Teuchos::Exceptions::InvalidParameter, "Aeras::ShallowWaterResid must be run such that nNodes == numQPs!  "
			       <<  "This does not hold: numNodes = " <<  nNodes << ", numQPs = " << numQPs << ".");
  }

  this->setName("Aeras::ShallowWaterResid"+PHX::typeAsString<EvalT>());

  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  std::vector<PHX::DataLayout::size_type> gradDims;
  wGradBF.fieldTag().dataLayout().dimensions(gradDims);
  GradBF.fieldTag().dataLayout().dimensions(gradDims);

  //std::cout << " vecDim = " << vecDim << std::endl;
  //std::cout << " numDims = " << numDims << std::endl;
  //std::cout << " numQPs = " << numQPs << std::endl;
  //std::cout << " numNodes = " << numNodes << std::endl;

  // Register Reynolds number as Sacado-ized Parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
  this->registerSacadoParameter("Gravity", paramLib);
  this->registerSacadoParameter("Omega", paramLib);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
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

#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::postRegistrationSetup" << std::endl;
#endif

  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UNodal,fm);
  this->utils.setFieldData(UDot,fm);
  //OG UDotDot is not in use?
  /*
  this->utils.setFieldData(UDotDot,fm);
  this->utils.setFieldData(UDotDotNodal,fm);
  */
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(mountainHeight,fm);

  this->utils.setFieldData(sphere_coord,fm);
  /*
  this->utils.setFieldData(hyperviscosity,fm);
  */
  this->utils.setFieldData(lambda_nodal,fm);
  this->utils.setFieldData(theta_nodal,fm);
  /*
  this->utils.setFieldData(source,fm);
  */
  this->utils.setFieldData(weighted_measure, fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_inv, fm);
  this->utils.setFieldData(jacobian_det, fm);

  this->utils.setFieldData(Residual,fm);

  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPs, 2);
  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, 2);

  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  nodal_jacobian = Kokkos::createDynRankView(wBF.get_view(), "XXX", numNodes, 2, 2);
  nodal_inv_jacobian = Kokkos::createDynRankView(wBF.get_view(), "XXX", numNodes, 2, 2);
  nodal_det_j = Kokkos::createDynRankView(wBF.get_view(), "XXX", numNodes);
  wrk_ = Kokkos::createDynRankView(U.get_view(), "XXX", numNodes, 2);

#else
  //Allocationg additional data for Kokkos functors
  refWeights_Kokkos=Kokkos::createDynRankView(wBF.get_view(),"refWeights_Kokkos",numQPs);
  grad_at_cub_points_Kokkos=Kokkos::createDynRankView(wBF.get_view(),"grad_at_cub_points_Kokkos",numNodes,numQPs,2);
  refPoints_kokkos=Kokkos::createDynRankView(wBF.get_view(),"refPoints_Kokkos",numQPs,2);

  for (int i=0; i<numQPs; i++) {
    refWeights_Kokkos(i)=refWeights(i);
    for (int j=0; j<2; j++) {
      refPoints_kokkos(i,j)=refPoints(i,j);
      for (int k=0; k<numNodes; k++)
        grad_at_cub_points_Kokkos(k,i,j)=grad_at_cub_points(k,i,j);
    }
  }

  //OG Later I will insert if-statements because some of these temporary fields do not need to be constructed
  //if HV is off.
  /*
  csurf = Kokkos::createDynRankView(Residual.get_view(), "csurf", numCells, numNodes);
  csurftilde = Kokkos::createDynRankView(Residual.get_view(), "csurftilde", numCells, numNodes);
  cgradsurf = Kokkos::createDynRankView(Residual.get_view(), "cgradsurf", numCells, numQPs, 2);
  cgradsurftilde = Kokkos::createDynRankView(Residual.get_view(), "cgradsurftilde", numCells, numQPs, 2);
  */
  tempnodalvec1 = Kokkos::createDynRankView(Residual.get_view(), "tempnodalvec1", numCells, numNodes, 2);
  tempnodalvec2 = Kokkos::createDynRankView(Residual.get_view(), "tempnodalvec2", numCells, numNodes, 2);
  /*
  chuv = Kokkos::createDynRankView(Residual.get_view(), "chuv", numCells, numNodes, 2);
  cdiv = Kokkos::createDynRankView(Residual.get_view(), "cdiv", numCells, numQPs);
  ccor = Kokkos::createDynRankView(Residual.get_view(), "ccor", numCells, numQPs);
  */
  ckineticEnergy = Kokkos::createDynRankView(Residual.get_view(), "ckineticEnergy", numCells, numNodes);
  cpotentialEnergy = Kokkos::createDynRankView(Residual.get_view(), "cpotentialEnergy", numCells, numNodes);
  /*
  cvelocityVec = Kokkos::createDynRankView(Residual.get_view(), "cvelocityVec", numCells, numNodes, 2);
  cvort = Kokkos::createDynRankView(Residual.get_view(), "cvort", numCells, numQPs);
  cgradKineticEnergy = Kokkos::createDynRankView(Residual.get_view(), "cgradKineticEnergy", numCells, numQPs, 2);
  cgradPotentialEnergy = Kokkos::createDynRankView(Residual.get_view(), "cgradPotentialEnergy", numCells, numQPs, 2);
  cUX = Kokkos::createDynRankView(Residual.get_view(), "cUX", numCells, numNodes);
  cUY = Kokkos::createDynRankView(Residual.get_view(), "cUY", numCells, numNodes);
  cUZ = Kokkos::createDynRankView(Residual.get_view(), "cUZ", numCells, numNodes);
  cUTX = Kokkos::createDynRankView(Residual.get_view(), "cUTX", numCells, numNodes);
  cUTY = Kokkos::createDynRankView(Residual.get_view(), "cUTY", numCells, numNodes);
  cUTZ = Kokkos::createDynRankView(Residual.get_view(), "cUTZ", numCells, numNodes);
  cgradUX = Kokkos::createDynRankView(Residual.get_view(), "cgradUX", numCells, numQPs, 2);
  cgradUY = Kokkos::createDynRankView(Residual.get_view(), "cgradUY", numCells, numQPs, 2);
  cgradUZ = Kokkos::createDynRankView(Residual.get_view(), "cgradUZ", numCells, numQPs, 2);
  cgradUTX = Kokkos::createDynRankView(Residual.get_view(), "cgradUTX", numCells, numQPs, 2);
  cgradUTY = Kokkos::createDynRankView(Residual.get_view(), "cgradUTY", numCells, numQPs, 2);
  cgradUTZ = Kokkos::createDynRankView(Residual.get_view(), "cgradUTZ", numCells, numQPs, 2);
  */
#endif
}

// *********************************************************************
//Kokkos functors
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
/*
template< typename ScalarT, typename ArrayT1,typename ArrayT2, typename ArrayJac, typename ArrayGrad>
KOKKOS_INLINE_FUNCTION
void gradient(const ArrayT1  & fieldAtNodes, const int &cell, ArrayT2  & gradField, 
              ArrayJac &jacobian_inv, ArrayGrad &grad_at_cub_points_Kokkos) 
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::gradient (kokkos)" << std::endl;
#endif
  for (int qp=0; qp < grad_at_cub_points_Kokkos.dimension(1); ++qp) {
    ScalarT gx = 0;
    ScalarT gy = 0;
    for (int node=0; node < grad_at_cub_points_Kokkos.dimension(0); ++node) {
      const typename PHAL::Ref<const ScalarT>::type field = fieldAtNodes(node);
      gx +=   field*grad_at_cub_points_Kokkos(node, qp,0);
      gy +=   field*grad_at_cub_points_Kokkos(node, qp,1);
    }
    gradField(qp, 0) = jacobian_inv(cell, qp, 0, 0)*gx + jacobian_inv(cell, qp, 1, 0)*gy;
    gradField(qp, 1) = jacobian_inv(cell, qp, 0, 1)*gx + jacobian_inv(cell, qp, 1, 1)*gy;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::divergence4(const Kokkos::DynRankView<ScalarT, PHX::Device>  & fieldAtNodes,
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & div_,
  const int & cell) const  
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::divergence4 (kokkos)" << std::endl;
#endif

  for (std::size_t node=0; node < numNodes; ++node) {
    const MeshScalarT jinv00 = jacobian_inv(cell, node, 0, 0);
    const MeshScalarT jinv01 = jacobian_inv(cell, node, 0, 1);
    const MeshScalarT jinv10 = jacobian_inv(cell, node, 1, 0);
    const MeshScalarT jinv11 = jacobian_inv(cell, node, 1, 1);
    const MeshScalarT det_j  = jacobian_det(cell,node);

    // constructing contravariant velocity
    tempnodalvec1(cell, node, 0 ) = det_j*(jinv00*fieldAtNodes(cell, node, 0) + jinv01*fieldAtNodes(cell, node, 1) );
    tempnodalvec1(cell, node, 1 ) = det_j*(jinv10*fieldAtNodes(cell, node, 0) + jinv11*fieldAtNodes(cell, node, 1) );
  }

  for (int qp=0; qp < numQPs; ++qp) {
    div_(cell, qp) = 0.0;
    for (int node=0; node < numNodes; ++node) {
      //OG What is this commented code?
      //ScalarT tempAdd =vcontra(node, 0)*grad_at_cub_points_Kokkos(node, qp,0)
      //                + vcontra(node, 1)*grad_at_cub_points_Kokkos(node, qp,1);
      //     Kokkos::atomic_fetch_add(&div_hU(qp), tempAdd);
      div_(cell, qp) += tempnodalvec1(cell, node, 0)*grad_at_cub_points_Kokkos(node, qp, 0)
                     +  tempnodalvec1(cell, node, 1)*grad_at_cub_points_Kokkos(node, qp, 1);
    }
  }

  for (int qp=0; qp < numQPs; ++qp) {
    div_(cell, qp) = div_(cell, qp)/jacobian_det(cell,qp);
  }

}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::
gradient4(const Kokkos::DynRankView<ScalarT, PHX::Device>  & field,
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & gradient_,
  const int & cell) const 
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::gradient4 (kokkos)" << std::endl;
#endif

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    ScalarT gx = 0;
    ScalarT gy = 0;
    for (std::size_t node=0; node < numNodes; ++node) {
      //const typename PHAL::Ref<const ScalarT>::type
      const ScalarT field_ = field(cell,node);
      gx += field_*grad_at_cub_points_Kokkos(node, qp,0);
      gy += field_*grad_at_cub_points_Kokkos(node, qp,1);
    }

    gradient_(cell,qp, 0) = jacobian_inv(cell, qp, 0, 0)*gx + jacobian_inv(cell, qp, 1, 0)*gy;
    gradient_(cell,qp, 1) = jacobian_inv(cell, qp, 0, 1)*gx + jacobian_inv(cell, qp, 1, 1)*gy;
  }

}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::curl4(
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & field,
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & curl_,
  const int &cell) const 
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::curl4 (kokkos)" << std::endl;
#endif
  for (int node=0; node < numNodes; ++node) {
    const MeshScalarT j00 = jacobian(cell, node, 0, 0);
    const MeshScalarT j01 = jacobian(cell, node, 0, 1);
    const MeshScalarT j10 = jacobian(cell, node, 1, 0);
    const MeshScalarT j11 = jacobian(cell, node, 1, 1);

    //forming covariant vector
    tempnodalvec2(cell, node, 0 ) = j00*field(cell, node, 0) + j10*field(cell, node, 1);
    tempnodalvec2(cell, node, 1 ) = j01*field(cell, node, 0) + j11*field(cell, node, 1);
  }
  for (int qp=0; qp < numQPs; ++qp) {
    curl_(cell, qp) = 0.0;
    for (int node=0; node < numNodes; ++node) {
      curl_(cell, qp) += tempnodalvec2(cell, node, 1)*grad_at_cub_points_Kokkos(node, qp, 0)
      		      -  tempnodalvec2(cell, node, 0)*grad_at_cub_points_Kokkos(node, qp, 1);
    }
    curl_(cell, qp) = curl_(cell, qp)/jacobian_det(cell, qp);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT,typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT,Traits>::
get_coriolis4(const Kokkos::DynRankView<ScalarT, PHX::Device>  & cor_,
  const int &cell) const 
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::get_coriolis4 (kokkos)" << std::endl;
#endif

  //double alpha = AlphaAngle; //must match what is in initial condition for TC2 and TC5.
  //see AAdatpt::AerasZonal analytic function.
  //
  for (int qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    cor_(cell,qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(AlphaAngle) + sin(theta)*cos(AlphaAngle));
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residual0(const int& cell) const
{

#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_Residual0 (kokkos)" << std::endl;
#endif

  //OG This is where to my best knowledge code with kossos and FPE check crashes.
  //So print statements are left.
  for (int node=0; node < numNodes; ++node) {
    //const ScalarT
    //unodal0 = UNodal(cell,node,0);
    const typename PHAL::Ref<const ScalarT>::type
    unodal0 = UNodal(cell,node,0);
*/
/*   std::cout << "ShallowWaterResid::compute_Residual0  inside loop 2 after assign unodal0"  << std::endl;
    std::cout << "address chuv(cell,node,0)"  << chuv(cell,node,0) << std::endl;
    std::cout << "address UNodal(cell,node,1)"  << UNodal(cell,node,1) << std::endl;
    std::cout << "address unodal0"  << unodal0 << std::endl;
    std::cout << "address UNodal(cell,node,2)"  << UNodal(cell,node,2) << std::endl;
    std::cout << "address chuv(cell,node,1)"  << chuv(cell,node,1) << std::endl; 
*/
/*
    chuv(cell,node,0) = unodal0*UNodal(cell,node,1);
    chuv(cell,node,1) = unodal0*UNodal(cell,node,2);
//    std::cout << "ShallowWaterResid::compute_Residual0  inside loop 3 after assign chuv"  << std::endl;

  }

//  std::cout << "ShallowWaterResid::compute_Residual0 before div4"  << std::endl;
  divergence4(chuv, cdiv, cell);
//  std::cout << "ShallowWaterResid::compute_Residual0 after div4"  << std::endl;
  for (int qp=0; qp < numQPs; ++qp) {
//    std::cout << "ShallowWaterResid::compute_Residual0 in loop before resid assignment"  << std::endl;
    int node = qp;
    Residual(cell,node,0) += (UDot(cell,qp,0) + cdiv(cell,qp))*wBF(cell, node, qp);
//    std::cout << "ShallowWaterResid::compute_Residual0 in loop after resid assignment"  << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_h_ImplHV(const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_h_ImplHV (kokkos)" << std::endl;
#endif

  for (std::size_t node=0; node < numNodes; ++node)
    csurftilde(cell,node) = UNodal(cell,node,3);

  gradient4(csurftilde, cgradsurftilde, cell);

  for (int qp=0; qp < numQPs; ++qp) {
    for (int node=0; node < numNodes; ++node) {
      Residual(cell,node,0) -= hyperviscosity(cell,qp,0)*cgradsurftilde(cell,qp,0)*wGradBF(cell,node,qp,0)
			    + hyperviscosity(cell,qp,0)*cgradsurftilde(cell,qp,1)*wGradBF(cell,node,qp,1);
    }
  }

  for (std::size_t node=0; node < numNodes; ++node)
    csurf(cell,node) = UNodal(cell,node,0);

  gradient4(csurf, cgradsurf, cell);

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    size_t node = qp;
    Residual(cell,node,3) += U(cell,qp,3)*wBF(cell,node,qp);
  }
  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t node=0; node < numNodes; ++node) {
      Residual(cell,node,3) += cgradsurf(cell,qp,0)*wGradBF(cell,node,qp,0)
                      	    +  cgradsurf(cell,qp,1)*wGradBF(cell,node,qp,1);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
zeroing_Residual(const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::zeroing_Residual (kokkos)" << std::endl;
#endif
  for(std::size_t node=0; node < numNodes; ++node)
    for(std::size_t neq=0; neq < vecDim; ++neq)
      Residual(cell, node, neq) = 0.0;
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim3_usePrescribedVelocity_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_VecDim3_usePrescribedVelocity  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_Residuals12_prescribed(cell);
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim4_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_VecDim4  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_h_ImplHV(cell);
  compute_Residuals12_prescribed(cell);
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residuals12_prescribed(const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_Residuals12_prescribed  (kokkos)" << std::endl;
#endif

  for (int qp=0; qp < numQPs; ++qp) {
    // before this loop used int node = qp;
    //  ... wBF(cell, node, qp), ...Residual(cell,node,1)

    //const ScalarT
    //wbf = wBF(cell, qp, qp);

    //OG Something should be done about this source,
    //it is there for TC4 and is hardly in use (not to mention TC4 is broken).
    //Residual(cell,qp,1) += (UDot(cell,qp,1) + source(cell,qp,1))*wbf;
    //Residual(cell,qp,2) += (UDot(cell,qp,2) + source(cell,qp,2))*wbf;

    Residual(cell,qp,1) += (UDot(cell,qp,1) + source(cell,qp,1))*wBF(cell, qp, qp);
    Residual(cell,qp,2) += (UDot(cell,qp,2) + source(cell,qp,2))*wBF(cell, qp, qp);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residuals12_notprescribed (const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_Residuals12_notprescribed  (kokkos)" << std::endl;
#endif
  get_coriolis4(ccor, cell);

  for (int node=0; node < numNodes; ++node) {
    //const typename PHAL::Ref<const ScalarT>::type
*/
    /*const ScalarT
    depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]),
    ulambda = UNodal(cell, node,1),
    utheta  = UNodal(cell, node,2);

    ckineticEnergy(cell, node) = 0.5*(ulambda*ulambda + utheta*utheta);
    cpotentialEnergy(cell, node) = gravity*depth;

    cvelocityVec(cell, node, 0) = ulambda;
    cvelocityVec(cell, node, 1) = utheta;
    */
/*

    ckineticEnergy(cell, node) = 0.5*(UNodal(cell, node,1)*UNodal(cell, node,1) + UNodal(cell, node,2)*UNodal(cell, node,2));
    cpotentialEnergy(cell, node) = gravity*(UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]));

    cvelocityVec(cell, node, 0) = UNodal(cell, node,1);
    cvelocityVec(cell, node, 1) = UNodal(cell, node,2);
  }

  curl4(cvelocityVec, cvort, cell);

  gradient4(ckineticEnergy, cgradKineticEnergy, cell);
  gradient4(cpotentialEnergy, cgradPotentialEnergy, cell);

  for (int qp=0; qp < numQPs; ++qp) {
    //int node = qp;
    const typename PHAL::Ref<const ScalarT>::type
    coriolis_ = ccor(cell, qp),
    curl_ = cvort(cell, qp),//  old code curl_ = curlU(qp)
    wbf_ = wBF(cell, qp, qp);

    Residual(cell,qp,1) += (   UDot(cell,qp,1) + cgradKineticEnergy(cell, qp, 0)
			+ cgradPotentialEnergy(cell, qp, 0)
			- ( coriolis_ + curl_ )*U(cell, qp, 2))*wbf_;
    Residual(cell,qp,2) += (   UDot(cell,qp,2) + cgradKineticEnergy(cell, qp, 1)
			+ cgradPotentialEnergy(cell, qp, 1)
			+ ( coriolis_ + curl_ )*U(cell, qp, 1))*wbf_;
  }
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_Residuals12_Vorticity_notprescribed (const int& cell, const int& index) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_Residuals12_Vorticity_notprescribed  (kokkos)" << std::endl;
#endif
  get_coriolis4(ccor, cell);

  for (int node=0; node < numNodes; ++node) {
    //const typename PHAL::Ref<const ScalarT>::type
*/
    /*const ScalarT
    depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]),
    ulambda = UNodal(cell, node,1),
    utheta  = UNodal(cell, node,2);

    ckineticEnergy(cell, node) = 0.5*(ulambda*ulambda + utheta*utheta);
    cpotentialEnergy(cell, node) = gravity*depth;

    cvelocityVec(cell, node, 0) = ulambda;
    cvelocityVec(cell, node, 1) = utheta;
    */
/*

    ckineticEnergy(cell, node) = 0.5*(UNodal(cell, node,1)*UNodal(cell, node,1) + UNodal(cell, node,2)*UNodal(cell, node,2));
    cpotentialEnergy(cell, node) = gravity*(UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]));

    cvelocityVec(cell, node, 0) = UNodal(cell, node,1);
    cvelocityVec(cell, node, 1) = UNodal(cell, node,2);
  }

  curl4(cvelocityVec, cvort, cell);

  gradient4(ckineticEnergy, cgradKineticEnergy, cell);
  gradient4(cpotentialEnergy, cgradPotentialEnergy, cell);

  for (int qp=0; qp < numQPs; ++qp) {
    //int node = qp;
    const typename PHAL::Ref<const ScalarT>::type
    coriolis_ = ccor(cell, qp),
    curl_ = cvort(cell, qp),//  old code curl_ = curlU(qp)
    wbf_ = wBF(cell, qp, qp);

    Residual(cell,qp,1) += (   UDot(cell,qp,1) + cgradKineticEnergy(cell, qp, 0)
			+ cgradPotentialEnergy(cell, qp, 0)
			- ( coriolis_ + curl_ )*U(cell, qp, 2))*wbf_;
    Residual(cell,qp,2) += (   UDot(cell,qp,2) + cgradKineticEnergy(cell, qp, 1)
			+ cgradPotentialEnergy(cell, qp, 1)
			+ ( coriolis_ + curl_ )*U(cell, qp, 1))*wbf_;
    //Vorticity
    Residual(cell,qp,index) += (U(cell,qp,index) - curl_)*wbf_;
  }

}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_VecDim3_no_usePrescribedVelocity  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_Residuals12_notprescribed(cell);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim3_Vorticity_no_usePrescribedVelocity_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_VecDim3_Vorticity_no_usePrescribedVelocity  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_Residuals12_Vorticity_notprescribed(cell, 3);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_BuildLaplace_for_huv_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_BuildLaplace_for_huv  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  BuildLaplace_for_h(cell);
  BuildLaplace_for_uv(cell);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_BuildLaplace_for_huv_Vorticity_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_BuildLaplace_for_huv_Vorticity  (kokkos)" << std::endl;
#endif

  setVecDim3_for_Vorticity(cell); 
}



///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_3Dvelocity4(std::size_t node, const ScalarT lam, const ScalarT th, const ScalarT ulambda, const ScalarT utheta,
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & ux, const Kokkos::DynRankView<ScalarT, PHX::Device>  & uy,
  const Kokkos::DynRankView<ScalarT, PHX::Device>  & uz, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_3Dvelocity4 (kokkos)" << std::endl;
#endif
  const ScalarT
  k11 = -sin(lam),
  k12 = -sin(th)*cos(lam),
  k21 =  cos(lam),
  k22 = -sin(th)*sin(lam),
  k32 =  cos(th);

  ux(cell, node) = k11*ulambda + k12*utheta;
  uy(cell, node) = k21*ulambda + k22*utheta;
  uz(cell, node) = k32*utheta;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
setVecDim3_for_Vorticity (const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::setVecDim3_for_Vorticity (kokkos)" << std::endl;
#endif
  for (std::size_t qp=0; qp < numQPs; ++qp)
    Residual(cell,qp,3) += UDot(cell,qp,3);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
BuildLaplace_for_uv (const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::BuildLaplace_for_uv (kokkos)" << std::endl;
#endif

  for (std::size_t node=0; node < numNodes; ++node) {

    const ScalarT ulambda = UDotDotNodal(cell, node,1),
                  utheta  = UDotDotNodal(cell, node,2);
    const ScalarT lam = lambda_nodal(cell, node),
		  th = theta_nodal(cell, node);

    compute_3Dvelocity4(node, lam, th, ulambda, utheta, cUX, cUY, cUZ, cell);

  }

  gradient4(cUX, cgradUX, cell);
  gradient4(cUY, cgradUY, cell);
  gradient4(cUZ, cgradUZ, cell);

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t node=0; node < numNodes; ++node) {
      //const typename PHAL::Ref<const ScalarT>::type
      const ScalarT lam = sphere_coord(cell, node, 0),
		    th  = sphere_coord(cell, node, 1),
		    wgradbf0_ = wGradBF(cell, node, qp, 0),
		    wgradbf1_ = wGradBF(cell, node, qp, 1);
      //K = -sin L    -sin T cos L
      //     cos L    -sin T sin L
      //     0         cos T
      //K^{-1} = K^T
      const ScalarT k11 = -sin(lam),
		    k12 = -sin(th)*cos(lam),
		    k21 =  cos(lam),
		    k22 = -sin(th)*sin(lam),
		    k32 =  cos(th);

      Residual(cell,node,1) +=	sHvTau*(k11*( cgradUX(cell, qp, 0)*wgradbf0_ + cgradUX(cell, qp, 1)*wgradbf1_ )
			    + k21*( cgradUY(cell, qp, 0)*wgradbf0_ + cgradUY(cell, qp, 1)*wgradbf1_ )
      //k31 = 0
    			    );

      Residual(cell,node,2) += sHvTau*(k12*( cgradUX(cell, qp, 0)*wgradbf0_ + cgradUX(cell, qp, 1)*wgradbf1_)
			    + k22*( cgradUY(cell, qp, 0)*wgradbf0_ + cgradUY(cell, qp, 1)*wgradbf1_)
			    + k32*( cgradUZ(cell, qp, 0)*wgradbf0_ + cgradUZ(cell, qp, 1)*wgradbf1_));
*/
      /*
      if (doNotDampRotation) {
        //adding back the first mode (in sph. harmonic basis) which corresponds to -2/R/R eigenvalue of laplace

        Residual(cell,node,1) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,4)*RRadius*RRadius*wBF(cell,node,qp);

        Residual(cell,node,2) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,5)*RRadius*RRadius*wBF(cell,node,qp);

        Residual(cell,node,4) += -2.0*U(cell,qp,1)*wBF(cell,node,qp)*RRadius*RRadius;

        Residual(cell,node,5) += -2.0*U(cell,qp,2)*wBF(cell,node,qp)*RRadius*RRadius;
      } */
/*
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_BuildLaplace_for_h_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() tag ShallowWaterResid_BuildLaplace_for_h (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  BuildLaplace_for_h(cell);
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
BuildLaplace_for_h (const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::BuildLaplace_for_h (kokkos)" << std::endl;
#endif

  //laplace forming for h field

  for (std::size_t node=0; node < numNodes; ++node)
    csurf(cell,node) = UDotDotNodal(cell,node,0);

  gradient4(csurf, cgradsurf, cell);

  for (std::size_t qp=0; qp < numQPs; ++qp) {
    for (std::size_t node=0; node < numNodes; ++node) {
      Residual(cell,node,0) += sHvTau*cgradsurf(cell,qp,0)*wGradBF(cell,node,qp,0)
                            +  sHvTau*cgradsurf(cell,qp,1)*wGradBF(cell,node,qp,1);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim6_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() ShallowWaterResid_VecDim6 tag  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_h_ImplHV(cell);
  compute_Residuals12_notprescribed(cell);
  compute_uv_ImplHV(cell);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_VecDim6_Vorticity_Tag& tag, const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::() ShallowWaterResid_VecDim6_Vorticity tag  (kokkos)" << std::endl;
#endif

  zeroing_Residual(cell);

  compute_Residual0(cell);
  compute_h_ImplHV(cell);
  compute_Residuals12_Vorticity_notprescribed(cell, 6);
  compute_uv_ImplHV(cell);
}


///////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
compute_uv_ImplHV (const int& cell) const
{
#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::compute_uv_ImplHV (kokkos)" << std::endl;
#endif

  for (int node=0; node < numNodes; ++node) {
    const ScalarT ulambda = UNodal(cell, node,1),
		  utheta  = UNodal(cell, node,2),
		  utlambda = UNodal(cell, node,4),
		  uttheta = UNodal(cell, node,5);

    const ScalarT lam = lambda_nodal(cell, node),
		  th = theta_nodal(cell, node);

    compute_3Dvelocity4(node, lam, th, ulambda, utheta, cUX, cUY, cUZ, cell);
    compute_3Dvelocity4(node, lam, th, utlambda, uttheta, cUTX, cUTY, cUTZ, cell);
  }

  gradient4(cUX, cgradUX, cell);
  gradient4(cUY, cgradUY, cell);
  gradient4(cUZ, cgradUZ, cell);

  gradient4(cUTX, cgradUTX, cell);
  gradient4(cUTY, cgradUTY, cell);
  gradient4(cUTZ, cgradUTZ, cell);

  //OG It seems that reversing these loops (nodal first, qp second)
  //will make it more efficient because of lam, th, weights assignments. Something to consider.
  for (int qp=0; qp < numQPs; ++qp) {
    const ScalarT wbf_ = wBF(cell,qp,qp); 
    for (int node=0; node < numNodes; ++node) {
      const ScalarT lam = sphere_coord(cell, node, 0),
		    th  = sphere_coord(cell, node, 1),
		    wgradbf0_ = wGradBF(cell, node, qp, 0),
		    wgradbf1_ = wGradBF(cell, node, qp, 1);
      //K = -sin L    -sin T cos L
      //     cos L    -sin T sin L
      //     0         cos T
      //K^{-1} = K^T

      const ScalarT k11 = -sin(lam),
		    k12 = -sin(th)*cos(lam),
		    k21 =  cos(lam),
		    k22 = -sin(th)*sin(lam),
		    k32 =  cos(th);

      //compute_coefficients_K(lam,th);
      //Do not delete:
      //Consider
      //V - tensor in tensor HV formulation, not hyperviscosity coefficient,
      //assume V = [v11 v12; v21 v22] then expressions below, for Residual(cell,node,1)
      //would take form
      //     k11*( (v11*utXgradNodes(qp,0) + v12*utXgradNodes(qp,1))*wGradBF(cell,node,qp,0) +
      //             (v21*utXgradNodes(qp,0) + v22*utXgradNodes(qp,1))*wGradBF(cell,node,qp,1)
      //           )
      //     + k21*( (v11*utYgradNodes(qp,0) + v12*utYgradNodes(qp,1))*wGradBF(cell,node,qp,0) +
      //             (v21*utYgradNodes(qp,0) + v22*utYgradNodes(qp,1))*wGradBF(cell,node,qp,1)
      //           )
      //instead of
      //Residual(cell,node,1) -=
      //		hyperviscosity(cell,qp,0)*(
      //				k11*( utXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
      //				+ k21*( utYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
      //				//k31 = 0
      //		);

      Residual(cell,node,1) -= hyperviscosity(cell,qp,0)*(
			       k11*( cgradUTX(cell, qp, 0)*wgradbf0_ + cgradUTX(cell, qp, 1)*wgradbf1_ )
			     + k21*( cgradUTY(cell, qp, 0)*wgradbf0_ + cgradUTY(cell, qp, 1)*wgradbf1_ )
			     //k31 = 0
			       );
      Residual(cell,node,2) -= hyperviscosity(cell,qp,0)*(
			       k12*( cgradUTX(cell, qp, 0)*wgradbf0_ + cgradUTX(cell, qp, 1)*wgradbf1_)
			     + k22*( cgradUTY(cell, qp, 0)*wgradbf0_ + cgradUTY(cell, qp, 1)*wgradbf1_)
			     + k32*( cgradUTZ(cell, qp, 0)*wgradbf0_ + cgradUTZ(cell, qp, 1)*wgradbf1_)
			       );

      Residual(cell,node,4) += k11*( cgradUX(cell, qp, 0)*wgradbf0_ + cgradUX(cell, qp, 1)*wgradbf1_ )
			    + k21*( cgradUY(cell, qp, 0)*wgradbf0_ + cgradUY(cell, qp, 1)*wgradbf1_ );
			    //k31 = 0

      Residual(cell,node,5) += k12*( cgradUX(cell, qp, 0)*wgradbf0_ + cgradUX(cell, qp, 1)*wgradbf1_)
			    + k22*( cgradUY(cell, qp, 0)*wgradbf0_ + cgradUY(cell, qp, 1)*wgradbf1_)
			    + k32*( cgradUZ(cell, qp, 0)*wgradbf0_ + cgradUZ(cell, qp, 1)*wgradbf1_);

    }
    if (doNotDampRotation) {
      //adding back the first mode (in sph. harmonic basis) which corresponds to -2/R/R eigenvalue of laplace
      Residual(cell,qp,1) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,4)*RRadius*RRadius*wbf_;
      Residual(cell,qp,2) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,5)*RRadius*RRadius*wbf_;
      Residual(cell,qp,4) += (U(cell,qp,4) - 2.0*U(cell,qp,1)*RRadius*RRadius)*wbf_;
      Residual(cell,qp,5) += (U(cell,qp,5) - 2.0*U(cell,qp,2)*RRadius*RRadius)*wbf_;
    }
  }
}
*/

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_TempNodalVec_Tag& tag, const int& cell, const int& node) const{
  // constructing contravariant velocity
  const MeshScalarT jinv00 = jacobian_inv(cell, node, 0, 0);
  const MeshScalarT jinv01 = jacobian_inv(cell, node, 0, 1);
  const MeshScalarT jinv10 = jacobian_inv(cell, node, 1, 0);
  const MeshScalarT jinv11 = jacobian_inv(cell, node, 1, 1);
  const MeshScalarT det_j  = jacobian_det(cell, node);

  const ScalarT unodal0 = UNodal(cell, node, 0);
  const ScalarT chuv0 = unodal0*UNodal(cell, node, 1);
  const ScalarT chuv1 = unodal0*UNodal(cell, node, 2);

  tempnodalvec1(cell, node, 0 ) = det_j*(jinv00*chuv0 + jinv01*chuv1 );
  tempnodalvec1(cell, node, 1 ) = det_j*(jinv10*chuv0 + jinv11*chuv1 );

  // forming covariant vector
  const MeshScalarT j00 = jacobian(cell, node, 0, 0);
  const MeshScalarT j01 = jacobian(cell, node, 0, 1);
  const MeshScalarT j10 = jacobian(cell, node, 1, 0);
  const MeshScalarT j11 = jacobian(cell, node, 1, 1);

  const ScalarT cvelocityVec0 = UNodal(cell, node, 1);
  const ScalarT cvelocityVec1 = UNodal(cell, node, 2);

  tempnodalvec2(cell, node, 0 ) = j00*cvelocityVec0 + j10*cvelocityVec1;
  tempnodalvec2(cell, node, 1 ) = j01*cvelocityVec0 + j11*cvelocityVec1;

  // compute energy
  ckineticEnergy(cell, node) = 0.5*(UNodal(cell, node,1)*UNodal(cell, node,1) + UNodal(cell, node,2)*UNodal(cell, node,2));
  cpotentialEnergy(cell, node) = gravity*(UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap_Kokkos[node]));
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ShallowWaterResid<EvalT, Traits>::
operator() (const ShallowWaterResid_Residual_Tag& tag, const int& cell, const int& qp) const{
  // Compute Coriolis
  const MeshScalarT lambda = sphere_coord(cell, qp, 0);
  const MeshScalarT theta = sphere_coord(cell, qp, 1);
  ScalarT cor = 2*Omega*( -cos(lambda)*cos(theta)*sin(AlphaAngle) + sin(theta)*cos(AlphaAngle));

  // Compute divergence of contravariant velocity
  ScalarT div = 0;
  for (int node=0; node < numNodes; ++node) {
    div += tempnodalvec1(cell, node, 0)*grad_at_cub_points_Kokkos(node, qp, 0)
        +  tempnodalvec1(cell, node, 1)*grad_at_cub_points_Kokkos(node, qp, 1);
  }
  div /= jacobian_det(cell, qp);

  // Compute curl of covariant vector
  ScalarT curl = 0;
  for (int node=0; node < numNodes; ++node) {
    curl += tempnodalvec2(cell, node, 1)*grad_at_cub_points_Kokkos(node, qp, 0)
         -  tempnodalvec2(cell, node, 0)*grad_at_cub_points_Kokkos(node, qp, 1);
  }
  curl /= jacobian_det(cell, qp);

  // Compute gradient of kinetic energy
  ScalarT gradKE[2] = {0};
  for (int node=0; node < numNodes; ++node) {
    gradKE[0] += ckineticEnergy(cell, node)*grad_at_cub_points_Kokkos(node, qp, 0);
    gradKE[1] += ckineticEnergy(cell, node)*grad_at_cub_points_Kokkos(node, qp, 1);
  }
  ScalarT gradKEtemp = gradKE[0];
  gradKE[0] = jacobian_inv(cell, qp, 0, 0)*gradKE[0]  + jacobian_inv(cell, qp, 1, 0)*gradKE[1];
  gradKE[1] = jacobian_inv(cell, qp, 0, 1)*gradKEtemp + jacobian_inv(cell, qp, 1, 1)*gradKE[1];

  // Compute gradient of potential energy
  ScalarT gradPE[2] = {0};
  for (int node=0; node < numNodes; ++node) {
    gradPE[0] += cpotentialEnergy(cell, node)*grad_at_cub_points_Kokkos(node, qp, 0);
    gradPE[1] += cpotentialEnergy(cell, node)*grad_at_cub_points_Kokkos(node, qp, 1);
  }
  ScalarT gradPEtemp = gradPE[0];
  gradPE[0] = jacobian_inv(cell, qp, 0, 0)*gradPE[0]  + jacobian_inv(cell, qp, 1, 0)*gradPE[1];
  gradPE[1] = jacobian_inv(cell, qp, 0, 1)*gradPEtemp + jacobian_inv(cell, qp, 1, 1)*gradPE[1];

  // Compute Residual
  Residual(cell, qp, 0) = (UDot(cell, qp, 0) + div)*wBF(cell, qp, qp);
  Residual(cell, qp, 1) = (UDot(cell, qp, 1) + gradKE[0] + gradPE[0] 
                        - (cor + curl) * U(cell, qp, 2)) * wBF(cell, qp, qp);
  Residual(cell, qp, 2) = (UDot(cell, qp, 2) + gradKE[1] + gradPE[1]
                        + (cor + curl) * U(cell, qp, 1)) * wBF(cell, qp, qp);
}

#endif
//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double j_coeff = workset.j_coeff;
  double m_coeff = workset.m_coeff;
  double n_coeff = workset.n_coeff;

  obtainLaplaceOp = (n_coeff == 1) ? true : false;

  //  MeshScalarT k11, k12, k21, k22, k32;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  PHAL::set(Residual, 0.0);
#ifdef ALBANY_VERBOSE
  std::cout << "In SW_resid: j_coeff, m_coeff, n_coeff: " << j_coeff << ", " << m_coeff << ", " << n_coeff << std::endl;
#endif 
  //Note that vars huAtNodes, div_hU, ... below are redefined locally here.
  //Global vars with such names exist too (see constructor).
  Kokkos::DynRankView<ScalarT, PHX::Device>  huAtNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes,2);
  Kokkos::DynRankView<ScalarT, PHX::Device>  div_hU = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs);
  Kokkos::DynRankView<ScalarT, PHX::Device>  div_weak_hU = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs);
  Kokkos::DynRankView<ScalarT, PHX::Device>  kineticEnergyAtNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device>  gradKineticEnergy = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device>  potentialEnergyAtNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device>  gradPotentialEnergy = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device>  uAtNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes, 2);
  Kokkos::DynRankView<ScalarT, PHX::Device>  curlU = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs);
  Kokkos::DynRankView<ScalarT, PHX::Device>  coriolis = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs);

  //container for surface height for viscosty
  Kokkos::DynRankView<ScalarT, PHX::Device> surf = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device> surftilde = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  //conteiner for surface height gradient for viscosity
  Kokkos::DynRankView<ScalarT, PHX::Device> hgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> htildegradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);

  //auxiliary vars, (u,v) in lon-lat is transformed to (ux,uy,uz) in XYZ
  Kokkos::DynRankView<ScalarT, PHX::Device> uX = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device> uY = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device> uZ = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);

  //auxiliary vars, (utilde,vtilde) in lon-lat is transformed to (utx,uty,utz) in XYZ
  Kokkos::DynRankView<ScalarT, PHX::Device> utX = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device> utY = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);
  Kokkos::DynRankView<ScalarT, PHX::Device> utZ = Kokkos::createDynRankView(U.get_view(),"ASW",numNodes);

  Kokkos::DynRankView<ScalarT, PHX::Device> uXgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> uYgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> uZgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> utXgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> utYgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);
  Kokkos::DynRankView<ScalarT, PHX::Device> utZgradNodes = Kokkos::createDynRankView(U.get_view(),"ASW",numQPs,2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    // Depth Equation (Eq# 0)
    Kokkos::deep_copy(huAtNodes,0.0);
    Kokkos::deep_copy(div_hU,0.0);
    Kokkos::deep_copy(surf,0.0);
    Kokkos::deep_copy(surftilde,0.0);
    Kokkos::deep_copy(hgradNodes,0.0);
    Kokkos::deep_copy(htildegradNodes,0.0);

    for (std::size_t node=0; node < numNodes; ++node) {
      ScalarT surfaceHeight = UNodal(cell,node,0);
      ScalarT ulambda = UNodal(cell, node,1);
      ScalarT utheta  = UNodal(cell, node,2);
      huAtNodes(node,0) = surfaceHeight*ulambda;
      huAtNodes(node,1) = surfaceHeight*utheta;
    }

    //OG this is a bug
    //for (std::size_t node=0; node < numNodes; ++node)
    //  surf(node) = UNodal(cell,node,0);
    //gradient(surf, cell, hgradNodes);


    //In case of Explicit Hyperviscosity we form Laplace operator if omega=n=1 .
    //This code should not be executed if hv coefficient is zero, the check
    //is in Albany_SolverFactory.

    if (useExplHyperviscosity) {
      //OG: maybe, int(n_coeff) == 1 ?
      if (n_coeff == 1) {  
        /*
	for (std::size_t node=0; node < numNodes; ++node)
  	  surftilde(node) = UDotDotNodal(cell,node,0);
      */
	gradient(surftilde, cell, htildegradNodes);

	for (std::size_t qp=0; qp < numQPs; ++qp) {
	  for (std::size_t node=0; node < numNodes; ++node) {
  	    Residual(cell,node,0) += sHvTau*htildegradNodes(qp,0)*wGradBF(cell,node,qp,0)
                       		  +  sHvTau*htildegradNodes(qp,1)*wGradBF(cell,node,qp,1);
	    //OG: This doesn't quite work when hvTau=0=hyperviscosity(:,:,:) .
	    //In case of hvTau = 0, sqrt(hyperviscosity(:))=[0 | nan nan ...] and laplace op. below contains nans as well.
	    //My best guess is that this is due to automatic differentiation.
	    //Residual(cell,node,0) += sqrt(hyperviscosity(cell,qp,0))*htildegradNodes(qp,0)*wGradBF(cell,node,qp,0)
	    //                      +  sqrt(hyperviscosity(cell,qp,0))*htildegradNodes(qp,1)*wGradBF(cell,node,qp,1);
  	  }
	}
      }//end if ncoeff==1

      //OG: this is a patch to fix vorticity field Residual(..,..,3)
      //for backward Euler. This adds a nontrivial block to the mass matrix that is stored to compute
      //a hyperviscosity  update for residual, LM^{-1}L. Since L contains zero block for vorticity
      //variable, the residual does not change with this modification. But nans in M^{-1} are avoided.
      //This if-statement may not be the best to detect the stage of computing M.
      if ((j_coeff == 0)&&(m_coeff == 1)&&(workset.current_time == 0)&&(plotVorticity)&&(!usePrescribedVelocity)) {
        for (std::size_t qp=0; qp < numQPs; ++qp) 
	  Residual(cell,qp,3) += UDot(cell,qp,3);
      }
    }//end of Laplace forming for h field

    if (useImplHyperviscosity) {
      for (std::size_t node=0; node < numNodes; ++node)
        surftilde(node) = UNodal(cell,node,3);
      gradient(surftilde, cell, htildegradNodes);
    }

#define WEAK_DIV 0
#if WEAK_DIV
    std::cout << "Weak divergence is on\n";
    fill_nodal_metrics(cell);
    Kokkos::deep_copy(div_weak_hU,0.0);

    // \int w_i * div(v) = - \sum_j \int grad(w_i)\cdot v_j w_j
    for (int i=0; i < numNodes; ++i)
      for (int j=0 ; j < numNodes; ++j) {
        div_weak_hU(i) -= huAtNodes(j,0) * GradBF(cell,i,j,0) * wBF(cell,j,j)
       		       +  huAtNodes(j,1) * GradBF(cell,i,j,1) * wBF(cell,j,j);
        //std::cout << "gradbf: " << cell << " " << node << " " << qp << " " << dim << " " << GradBF(cell,node,qp,dim) << std::endl;
	//std::cout << "val_node " << val_node(cell,node,level,dim) << std::endl;
      }
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      std::size_t node = qp;
      Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell, node, qp)
               		    +  div_weak_hU(qp);
    }
#else
    divergence(huAtNodes, cell, div_hU);
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      std::size_t node = qp;
      Residual(cell,node,0) += UDot(cell,qp,0)*wBF(cell, node, qp)
                      	    +  div_hU(qp)*wBF(cell, node, qp);
    }
#endif

    //OG This is for debugging, for the mesh with 24 total elements.
    /*
    if (cell == 23) {
      for (int node = 0; node < numNodes; node++) {
	std::cout << "QP = " << node <<"\n";
	std::cout <<"Metric term, jac_inv:"<< jacobian_inv(cell, node, 0, 0) <<" "<<jacobian_inv(cell, node, 0, 1) <<" \n";
	std::cout << "                      "<<jacobian_inv(cell, node, 1, 0) <<" "<<jacobian_inv(cell, node, 1, 1) <<" \n";
      }
    }
    */

    if (useImplHyperviscosity) { //hyperviscosity residual(0) = residual(0) - tau*grad(htilde)*grad(phi)
      //for tensor HV, hyperViscosity is (cell, qp, 2,2)
      //so the code below is temporary
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t node=0; node < numNodes; ++node) {
          /*
	  Residual(cell,node,0) -= hyperviscosity(cell,qp,0)*htildegradNodes(qp,0)*wGradBF(cell,node,qp,0)
                      		+  hyperviscosity(cell,qp,0)*htildegradNodes(qp,1)*wGradBF(cell,node,qp,1);
                          */
	}
      }
    }
  }

  if (useImplHyperviscosity) { //hyperviscosity residual(3) = htilde*phi + grad(h)*grad(phi)

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      Kokkos::deep_copy(surf,0.0);
      Kokkos::deep_copy(hgradNodes,0.0);
      for (std::size_t node=0; node < numNodes; ++node)
        surf(node) = UNodal(cell,node,0);
      gradient(surf, cell, hgradNodes);

      for (std::size_t qp=0; qp < numQPs; ++qp) {
        std::size_t node = qp;
	Residual(cell,node,3) += U(cell,qp,3)*wBF(cell,node,qp); 
      }
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	for (std::size_t node=0; node < numNodes; ++node) {
 	  Residual(cell,node,3) += hgradNodes(qp,0)*wGradBF(cell,node,qp,0)
               		        + hgradNodes(qp,1)*wGradBF(cell,node,qp,1);
	} 
      }
    }
  }// endif use hyperviscosity

  // Velocity Equations
  if (usePrescribedVelocity) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        std::size_t node = qp;
        /*
	Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp) + source(cell,qp,1)*wBF(cell, node, qp);
	Residual(cell,node,2) += UDot(cell,qp,2)*wBF(cell,node,qp) + source(cell,qp,2)*wBF(cell, node, qp);
  */
      }
    }
  }
  else { // Solve for velocity
    // Velocity Equations (Eq# 1,2)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      Kokkos::deep_copy(potentialEnergyAtNodes,0.0);
      Kokkos::deep_copy(gradPotentialEnergy,0.0);

      Kokkos::deep_copy(kineticEnergyAtNodes,0.0);
      Kokkos::deep_copy(gradKineticEnergy,0.0);
      Kokkos::deep_copy(uAtNodes,0.0);

      get_coriolis(cell, coriolis);

      if ((useExplHyperviscosity)&&(n_coeff == 1)) {
        //uX.initialize();
	//uY.initialize();
	//uZ.initialize();
	Kokkos::deep_copy(utX,0.0);
	Kokkos::deep_copy(utY,0.0);
	Kokkos::deep_copy(utZ,0.0);
	//uXgradNodes.initialize();
	//uYgradNodes.initialize();
	//uZgradNodes.initialize();
	Kokkos::deep_copy(utXgradNodes,0.0);
	Kokkos::deep_copy(utYgradNodes,0.0);
	Kokkos::deep_copy(utZgradNodes,0.0); 
      }
      if (useImplHyperviscosity) {
	Kokkos::deep_copy(uX,0.0);
	Kokkos::deep_copy(uY,0.0);
	Kokkos::deep_copy(uZ,0.0);
	Kokkos::deep_copy(utX,0.0);
	Kokkos::deep_copy(utY,0.0);
	Kokkos::deep_copy(utZ,0.0);
	Kokkos::deep_copy(uXgradNodes,0.0);
	Kokkos::deep_copy(uYgradNodes,0.0);
	Kokkos::deep_copy(uZgradNodes,0.0);
	Kokkos::deep_copy(utXgradNodes,0.0);
	Kokkos::deep_copy(utYgradNodes,0.0);
	Kokkos::deep_copy(utZgradNodes,0.0);
      }

      for (std::size_t node=0; node < numNodes; ++node) {
	ScalarT depth = UNodal(cell,node,0) + mountainHeight(cell, nodeToQPMap[node]);
	ScalarT ulambda = UNodal(cell, node,1);
	ScalarT utheta  = UNodal(cell, node,2);
	kineticEnergyAtNodes(node) = 0.5*(ulambda*ulambda + utheta*utheta);
	potentialEnergyAtNodes(node) = gravity*depth;
	uAtNodes(node, 0) = ulambda;
	uAtNodes(node, 1) = utheta;
	ScalarT utlambda;
	ScalarT uttheta;
	if ((useExplHyperviscosity)&&(n_coeff == 1)) {
    /*
	  utlambda = UDotDotNodal(cell, node,1);
	  uttheta  = UDotDotNodal(cell, node,2);
    */
	}

	if (useImplHyperviscosity) {
	  utlambda = UNodal(cell, node,4);
	  uttheta  = UNodal(cell, node,5);
	}

	if ((useExplHyperviscosity)&&(n_coeff == 1)) {
	  const typename PHAL::Ref<const MeshScalarT>::type lam = lambda_nodal(cell, node),
	                                                    th = theta_nodal(cell, node);
 	  const ScalarT	k11 = -sin(lam),
			k12 = -sin(th)*cos(lam),
			k21 =  cos(lam),
			k22 = -sin(th)*sin(lam),
			k32 =  cos(th);
 	  //uX(node) = k11*ulambda + k12*utheta;
	  //uY(node) = k21*ulambda + k22*utheta;
	  //uZ(node) = k32*utheta;
  	  utX(node) = k11*utlambda + k12*uttheta;
	  utY(node) = k21*utlambda + k22*uttheta;
	  utZ(node) = k32*uttheta;
	}

	if (useImplHyperviscosity) {
	  const typename PHAL::Ref<const MeshScalarT>::type lam = lambda_nodal(cell, node),
					                     th = theta_nodal(cell, node);
	  const ScalarT	k11 = -sin(lam),
			k12 = -sin(th)*cos(lam),
			k21 =  cos(lam),
			k22 = -sin(th)*sin(lam),
			k32 =  cos(th);

	  uX(node) = k11*ulambda + k12*utheta;
	  uY(node) = k21*ulambda + k22*utheta;
	  uZ(node) = k32*utheta;

	  utX(node) = k11*utlambda + k12*uttheta;
	  utY(node) = k21*utlambda + k22*uttheta;
	  utZ(node) = k32*uttheta;
	}
      }

      if ((useExplHyperviscosity)&&(n_coeff == 1)) {
	//gradient(uX, cell, uXgradNodes);
        //gradient(uY, cell, uYgradNodes);
	//gradient(uZ, cell, uZgradNodes);

	gradient(utX, cell, utXgradNodes);
	gradient(utY, cell, utYgradNodes);
	gradient(utZ, cell, utZgradNodes);
      }

      if (useImplHyperviscosity) {
	gradient(uX, cell, uXgradNodes);
	gradient(uY, cell, uYgradNodes);
	gradient(uZ, cell, uZgradNodes);
	gradient(utX, cell, utXgradNodes);
	gradient(utY, cell, utYgradNodes);
	gradient(utZ, cell, utZgradNodes);
      }

      gradient(potentialEnergyAtNodes, cell, gradPotentialEnergy);

      gradient(kineticEnergyAtNodes, cell, gradKineticEnergy);
      curl(uAtNodes, cell, curlU);

      if (plotVorticity) {
	if (useImplHyperviscosity) {
	  for (std::size_t qp=0; qp < numQPs; ++qp)
	    Residual(cell,qp,6) += (U(cell,qp,6) - curlU(qp))*wBF(cell,qp,qp);
	}
        else {
	  for (std::size_t qp=0; qp < numQPs; ++qp)
	    Residual(cell,qp,3) += (U(cell,qp,3) - curlU(qp))*wBF(cell,qp,qp);
	} 
      }

      for (std::size_t qp=0; qp < numQPs; ++qp) {
	size_t node = qp;
	Residual(cell,node,1) += (   UDot(cell,qp,1) + gradKineticEnergy(qp,0)
 			      + gradPotentialEnergy(qp,0)
			      - ( coriolis(qp) + curlU(qp) )*U(cell, qp, 2)
				)*wBF(cell,node,qp);
	Residual(cell,node,2) += (   UDot(cell,qp,2) + gradKineticEnergy(qp,1)
			      + gradPotentialEnergy(qp,1)
			      + ( coriolis(qp) + curlU(qp) )*U(cell, qp, 1)
				)*wBF(cell,node,qp);
      }
      if (useImplHyperviscosity) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
	  for (std::size_t node=0; node < numNodes; ++node) {
	    const typename PHAL::Ref<const MeshScalarT>::type lam = sphere_coord(cell, node, 0),
						               th = sphere_coord(cell, node, 1);

 	    //K = -sin L    -sin T cos L
	    //     cos L    -sin T sin L
	    //     0         cos T
	    //K^{-1} = K^T
	    const ScalarT k11 = -sin(lam),
			  k12 = -sin(th)*cos(lam),
			  k21 =  cos(lam),
			  k22 = -sin(th)*sin(lam),
			  k32 =  cos(th);
   	    //Do not delete:
	    //Consider
	    //V - tensor in tensor HV formulation, not hyperviscosity coefficient,
	    //assume V = [v11 v12; v21 v22] then expressions below, for Residual(cell,node,1)
	    //would take form
	    /*     k11*( (v11*utXgradNodes(qp,0) + v12*utXgradNodes(qp,1))*wGradBF(cell,node,qp,0) +
            (v21*utXgradNodes(qp,0) + v22*utXgradNodes(qp,1))*wGradBF(cell,node,qp,1)
            )
            + k21*( (v11*utYgradNodes(qp,0) + v12*utYgradNodes(qp,1))*wGradBF(cell,node,qp,0) + 
            (v21*utYgradNodes(qp,0) + v22*utYgradNodes(qp,1))*wGradBF(cell,node,qp,1)
            )
	    */
      /*
	    Residual(cell,node,1) -= hyperviscosity(cell,qp,0)*(
                                     k11*( utXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k21*( utYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   //k31 = 0
				   );
	    Residual(cell,node,2) -= hyperviscosity(cell,qp,0)*(
				     k12*( utXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k22*( utYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k32*( utZgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utZgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   );
  	    Residual(cell,node,4) += k11*( uXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + uXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				  + k21*( uYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + uYgradNodes(qp,1)*wGradBF(cell,node,qp,1));
				  //k31 = 0
	    Residual(cell,node,5) += k12*( uXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + uXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				  + k22*( uYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + uYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				  + k32*( uZgradNodes(qp,0)*wGradBF(cell,node,qp,0) + uZgradNodes(qp,1)*wGradBF(cell,node,qp,1));
          */

          }
	  if (doNotDampRotation) {
	    //adding back the first mode (in sph. harmonic basis) which corresponds to -2/R/R eigenvalue of laplace
      /*
	    Residual(cell,qp,1) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,4)*RRadius*RRadius*wBF(cell,qp,qp);
	    Residual(cell,qp,2) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,5)*RRadius*RRadius*wBF(cell,qp,qp);
	    Residual(cell,qp,4) += (U(cell,qp,4) - 2.0*U(cell,qp,1)*RRadius*RRadius)*wBF(cell,qp,qp);
	    Residual(cell,qp,5) += (U(cell,qp,5) - 2.0*U(cell,qp,2)*RRadius*RRadius)*wBF(cell,qp,qp);
      */
   	  }
	}
      }//end if ImplHV

      if ((useExplHyperviscosity)&&(n_coeff == 1)) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
	  for (std::size_t node=0; node < numNodes; ++node) {
	    const typename PHAL::Ref<const MeshScalarT>::type lam = sphere_coord(cell, node, 0),
						               th = sphere_coord(cell, node, 1);

	    //K = -sin L    -sin T cos L
	    //     cos L    -sin T sin L
	    //     0         cos T
	    //K^{-1} = K^T
	    const ScalarT k11 = -sin(lam),
			  k12 = -sin(th)*cos(lam),
			  k21 =  cos(lam),
			  k22 = -sin(th)*sin(lam),
			  k32 =  cos(th);

	    //Do not delete:
	    //Consider
	    //V - tensor in tensor HV formulation, not hyperviscosity coefficient,
	    //assume V = [v11 v12; v21 v22] then expressions below, for Residual(cell,node,1)
	    //would take form
	    /*     k11*( (v11*utXgradNodes(qp,0) + v12*utXgradNodes(qp,1))*wGradBF(cell,node,qp,0) +
                   (v21*utXgradNodes(qp,0) + v22*utXgradNodes(qp,1))*wGradBF(cell,node,qp,1)
                   )
                 + k21*( (v11*utYgradNodes(qp,0) + v12*utYgradNodes(qp,1))*wGradBF(cell,node,qp,0) +
                   (v21*utYgradNodes(qp,0) + v22*utYgradNodes(qp,1))*wGradBF(cell,node,qp,1)
                   )
	    */


	    Residual(cell,node,1) += sHvTau*(
				     k11*( utXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k21*( utYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   //k31 = 0
				   );
 	    Residual(cell,node,2) += sHvTau*(
				     k12*( utXgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utXgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k22*( utYgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utYgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   + k32*( utZgradNodes(qp,0)*wGradBF(cell,node,qp,0) + utZgradNodes(qp,1)*wGradBF(cell,node,qp,1))
				   );
 	    if(plotVorticity) Residual(cell,node,3) += 0.0;
	    /*
            if (doNotDampRotation) {
              //adding back the first mode (in sph. harmonic basis) which corresponds to -2/R/R eigenvalue of laplace

              Residual(cell,node,1) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,4)*RRadius*RRadius*wBF(cell,node,qp);
              Residual(cell,node,2) += -hyperviscosity(cell,qp,0)*2.0*U(cell,qp,5)*RRadius*RRadius*wBF(cell,node,qp);

              Residual(cell,node,4) += -2.0*U(cell,qp,1)*wBF(cell,node,qp)*RRadius*RRadius;

              Residual(cell,node,5) += -2.0*U(cell,qp,2)*wBF(cell,node,qp)*RRadius*RRadius;
            } */
	  }
	}
      }//end if ExplHV
    } // end workset cell loop
  } //end if !prescribedVelocities

#else
  a = Aeras::ShallowWaterConstants::self().earthRadius;
  myPi = Aeras::ShallowWaterConstants::self().pi;

  //Kokkos::deep_copy(Residual.get_view(), ScalarT(0.0));

/*
  if (usePrescribedVelocity) {
    if (useImplHyperviscosity)
      Kokkos::parallel_for(ShallowWaterResid_VecDim4_Policy(0,workset.numCells),*this);
    else if (useExplHyperviscosity) {
      if ( obtainLaplaceOp ) {
        Kokkos::parallel_for(ShallowWaterResid_BuildLaplace_for_h_Policy(0,workset.numCells),*this);
      }
      else
        Kokkos::parallel_for(ShallowWaterResid_VecDim3_usePrescribedVelocity_Policy(0,workset.numCells),*this);
    }
    else
      Kokkos::parallel_for(ShallowWaterResid_VecDim3_usePrescribedVelocity_Policy(0,workset.numCells),*this);
  }
  else {
    if (useImplHyperviscosity) {
      if (plotVorticity) 
        Kokkos::parallel_for(ShallowWaterResid_VecDim6_Vorticity_Policy(0,workset.numCells),*this); 
      else
        Kokkos::parallel_for(ShallowWaterResid_VecDim6_Policy(0,workset.numCells),*this);
    }
    else if (useExplHyperviscosity) {
      if ( obtainLaplaceOp ) {
        Kokkos::parallel_for(ShallowWaterResid_BuildLaplace_for_huv_Policy(0,workset.numCells),*this);
        if ((j_coeff == 0) && (m_coeff == 1) && (workset.current_time == 0) && (plotVorticity))
          Kokkos::parallel_for(ShallowWaterResid_BuildLaplace_for_huv_Vorticity_Policy(0,workset.numCells),*this);
      }
      else {
        if (plotVorticity)
          Kokkos::parallel_for(ShallowWaterResid_VecDim3_Vorticity_no_usePrescribedVelocity_Policy(0,workset.numCells),*this);
        else {
*/
          Kokkos::Experimental::md_parallel_for(ShallowWaterResid_TempNodalVec_Policy(
            {0,0},{(int)workset.numCells,(int)numNodes},ShallowWaterResid_TempNodalVec_TileSize),*this);
          Kokkos::Experimental::md_parallel_for(ShallowWaterResid_Residual_Policy(
            {0,0},{(int)workset.numCells,(int)numQPs},ShallowWaterResid_Residual_TileSize),*this);
          //Kokkos::parallel_for(ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Policy(0,workset.numCells),*this);
/*
        }
      }
    }
    else {
      if (plotVorticity)
        Kokkos::parallel_for(ShallowWaterResid_VecDim3_Vorticity_no_usePrescribedVelocity_Policy(0,workset.numCells),*this);
      else
        Kokkos::parallel_for(ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Policy(0,workset.numCells),*this);
    }
  }
*/

#ifdef AERAS_OUTPUT
  std::cout << "ShallowWaterResid::end of evaluateFields (kokkos)" << std::endl;
#endif
#endif
}

//**********************************************************************
// Provide Access to Parameter for sensitivity/optimization/UQ
template<typename EvalT,typename Traits>
typename ShallowWaterResid<EvalT,Traits>::ScalarT&
ShallowWaterResid<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="Gravity") return gravity;
//else if (n=="Omega") return Omega;
  return Omega;
}

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::divergence(const Kokkos::DynRankView<ScalarT, PHX::Device>  & fieldAtNodes,
		std::size_t cell, Kokkos::DynRankView<ScalarT, PHX::Device>  & div) {

  Kokkos::DynRankView<ScalarT, PHX::Device>& vcontra = wrk_;
  Kokkos::deep_copy(vcontra,0.0);

  fill_nodal_metrics(cell);

  Kokkos::deep_copy(vcontra,0.0);
  Kokkos::deep_copy(div,0.0);

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
      div(qp) += vcontra(node, 0)*grad_at_cub_points(node, qp,0)
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
#endif

//**********************************************************************
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::gradient(const Kokkos::DynRankView<ScalarT, PHX::Device>  & fieldAtNodes,
  std::size_t cell, Kokkos::DynRankView<ScalarT, PHX::Device>  & gradField) 
{
  Kokkos::deep_copy(gradField,0.0);

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
#endif

// *********************************************************************
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::fill_nodal_metrics(std::size_t cell) 
{
  Kokkos::deep_copy(nodal_jacobian,0.0);
  Kokkos::deep_copy(nodal_det_j,0.0);
  Kokkos::deep_copy(nodal_inv_jacobian,0.0);

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
#endif

// *********************************************************************
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
//og: rename this to vorticity
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::curl(const Kokkos::DynRankView<ScalarT, PHX::Device>  & nodalVector,
  std::size_t cell, Kokkos::DynRankView<ScalarT, PHX::Device>  & curl) 
{
  Kokkos::DynRankView<ScalarT, PHX::Device>& covariantVector = wrk_;
  Kokkos::deep_copy(covariantVector,0.0);

  fill_nodal_metrics(cell);

  Kokkos::deep_copy(covariantVector,0.0);
  Kokkos::deep_copy(curl,0.0);

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
      curl(qp) += covariantVector(node, 1)*grad_at_cub_points(node, qp,0)
                - covariantVector(node, 0)*grad_at_cub_points(node, qp,1);
    }
    curl(qp) = curl(qp)/jacobian_det(cell,qp);
  }

  /////////// Debugging option, to verufy 3d code
  /*if(cell == 0){
    Kokkos::DynRankView<ScalarT, PHX::Device>  dummy(numNodes,2);
    dummy.initialize();
    covariantVector.initialize();
    curl.initialize();
    for (std::size_t node=0; node < numNodes; ++node) {
      dummy(node,0) = node;
      dummy(node,1) = node;
    }
    for (std::size_t node=0; node < numNodes; ++node) {
      const MeshScalarT j00 = nodal_jacobian(node, 0, 0);
      const MeshScalarT j01 = nodal_jacobian(node, 0, 1);
      const MeshScalarT j10 = nodal_jacobian(node, 1, 0);
      const MeshScalarT j11 = nodal_jacobian(node, 1, 1);
      covariantVector(node, 0 ) = j00*dummy(node, 0) + j10*dummy(node, 1);
      covariantVector(node, 1 ) = j01*dummy(node, 0) + j11*dummy(node, 1);
    }

    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t node=0; node < numNodes; ++node) {
	curl(qp) += covariantVector(node, 1)*grad_at_cub_points(node, qp,0)
   		  - covariantVector(node, 0)*grad_at_cub_points(node, qp,1);
      }
      curl(qp) = curl(qp)/jacobian_det(cell,qp);
    }
    std::cout << "Vorticity DEBUG \n";
    for (std::size_t node=0; node < numNodes; ++node) {
      std::cout << "vort(" << node << ") = " << curl(node) <<"\n";
    }
  }*/

}
#endif

// *********************************************************************

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT,typename Traits>
void
ShallowWaterResid<EvalT,Traits>::get_coriolis(std::size_t cell, Kokkos::DynRankView<ScalarT, PHX::Device>  & coriolis) 
{
  Kokkos::deep_copy(coriolis,0.0);
  double alpha = AlphaAngle; /*must match what is in initial condition for TC2 and TC5.
  //see AAdatpt::AerasZonal analytic function. */
  for (std::size_t qp=0; qp < numQPs; ++qp) {
    const MeshScalarT lambda = sphere_coord(cell, qp, 0);
    const MeshScalarT theta = sphere_coord(cell, qp, 1);
    coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
  }
}
#endif


//**********************************************************************

}
