//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Sacado_Fad_Kokkos_ViewFactory.hpp"

#include "Albany_MeshSpecs.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "LandIce_ThicknessResidCell.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
ThicknessResidCell<EvalT, Traits>::
ThicknessResidCell(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  Hdiff    (p.get<std::string> ("Thickness Change Variable Name"), dl->node_scalar),
  H0       (p.get<std::string> ("Initial Thickness Name"), dl->node_scalar),
  coordVec (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector),
  Residual (p.get<std::string> ("Residual Name"), dl->node_scalar)
{
  //normals = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_lateral->qp_vector_spacedim);
  
  this->addDependentField(Hdiff);
  this->addDependentField(H0);
  this->addDependentField(coordVec);

  unsteady = p.get<bool>("Unsteady");
  if(unsteady) {
    dHdt = decltype(dHdt)(p.get<std::string> ("Thickness Dot Variable Name"), dl->node_scalar);
    this->addDependentField(dHdt);
  } 

  forcing = decltype(forcing)(p.get<std::string> ("Forcing Name"), dl->node_scalar);
  this->addDependentField(forcing);

  V = decltype(V)(p.get<std::string>("Velocity Name"),  dl->node_vector);
  this->addDependentField(V);

  this->addEvaluatedField(Residual);

  dt = p.get<Teuchos::RCP<double> >("Time Step Ptr");

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = p.get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");

  //for (const auto &ele : dl->side_layouts) 
  //  std::cout <<"Layout: " <<ele.first << std::endl;

  lateralSideSetName = p.isParameter("Lateral Side Set Name") ? p.get<std::string>("Lateral Side Set Name") : std::string("boundary_side_set_3");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(lateralSideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Lateral side data layout not found.\n");

  this->setName("ThicknessResidCell"+PHX::print<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  numVecFODims  = std::min(dims[2], PHX::DataLayout::size_type(2));

  dl->qp_gradient->dimensions(dims);
  cellDim = dims[2];

  const CellTopologyData * const elem_top = &meshSpecs->ctd;
  TEUCHOS_TEST_FOR_EXCEPTION (elem_top->dimension != 2, std::runtime_error,
                              "Error! This evaluator expects a 2D cell.\n");

  intrepidBasis = Albany::getIntrepid2Basis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology(elem_top));

  cubatureDegree = p.get<int>("Cubature Degree");
  numNodes = intrepidBasis->getCardinality();

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
#ifdef OUTPUT_TO_SCREEN
*out << " in LandIce Thickness residual! " << std::endl;
*out << " numNodes = " << numNodes << std::endl;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThicknessResidCell<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{
  physPointsCell = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, cellDim);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThicknessResidCell<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  // Initialize residual to 0.0
  Kokkos::deep_copy(Residual.get_view(), ScalarT(0.0));

  const Albany::SideSetList& ssList = *(workset.sideSets);
  std::map<const int, const int> elem_side_map;

  Albany::SideSetList::const_iterator it_latss = ssList.find(lateralSideSetName);
  if (it_latss != ssList.end()) {
    const std::vector<Albany::SideStruct>& latSideSet = it_latss->second;
    for (std::size_t iSide = 0; iSide < latSideSet.size(); ++iSide) {
      const int elem_LID = latSideSet[iSide].ws_elem_idx;
      const int elem_side = latSideSet[iSide].side_pos; 
      auto ret = elem_side_map.insert(std::pair<const int,const int>(elem_LID, elem_side));
      TEUCHOS_TEST_FOR_EXCEPTION ((ret.second==false) && (ret.first->second != elem_side), std::runtime_error, "Error! This evaluator does not support multiple sides associated to the same element.\n");
    }
  }

  //Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  if (cellDim==2) {
    //const std::vector<Albany::SideStruct>& sideSet = it_ss->second;

    Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basisGrad_refPointsSide;

    Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSideDet;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> invJacobianSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_gradBasis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> scratch;

    Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Points;
    Kokkos::DynRankView<ScalarT, PHX::Device> forcing_Points;
    Kokkos::DynRankView<ScalarT, PHX::Device> H_Points;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Points;

    Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Nodes;
    Kokkos::DynRankView<ScalarT, PHX::Device> forcing_Nodes;
    Kokkos::DynRankView<ScalarT, PHX::Device> H_Nodes;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Nodes;
    Kokkos::DynRankView<ScalarT, PHX::Device> gradH_Points;
    Kokkos::DynRankView<ScalarT, PHX::Device> divV_Points;

    // Loop over the sides that form the boundary condition
    for (std::size_t elem_LID = 0; elem_LID < workset.numCells; ++elem_LID) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      //const CellTopologyData_Subcell& side =  cellType->getCellTopologyData()->side[elem_side];
      unsigned int numSideNodes = cellType->getNodeCount();
      Intrepid2::DefaultCubatureFactory cubFactory;
      cubatureSide = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubatureDegree);
      unsigned int sideDims = cellType->getDimension();
      unsigned int numQPsSide = cubatureSide->getNumPoints();

      // Allocate Temporary Views (should be pre-allocated)
     // cubPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, sideDims);
      refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, cellDim);
      cubWeightsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide);
      basis_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide);
      basisGrad_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide, cellDim);

      jacobianSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      jacobianSideDet = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      invJacobianSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      weighted_measure = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      trans_basis_refPointsSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      trans_gradBasis_refPointsSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide, cellDim);
      //weighted_trans_basis_refPointsSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      scratch = Sacado::createDynRankView(jacobianSide,"XXS", numQPsSide*cellDim*cellDim);

      dHdt_Points = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      forcing_Points = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      H_Points = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      V_Points = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide, numVecFODims);

      // Pre-Calculate reference element quantities
      cubatureSide->getCubature(refPointsSide, cubWeightsSide);

      // Copy the coordinate data over to a temp container
     for (std::size_t node = 0; node < numNodes; ++node) {
       for (std::size_t dim = 0; dim < cellDim; ++dim)
         physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
         //physPointsCell(0, node, cellDim-1) = -1.0; //set z=-1 on internal cell nodes and z=0 side (see next lines).
     }
     //for (unsigned int i = 0; i < numSideNodes; ++i)
     //  physPointsCell(0, i, cellDim-1) = 1.0;  //set z=0 on side

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      //Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);
      //refPointsSide = cubPointsSide;

      //for (std::size_t node = 0; node < numNodes; ++node) {
      //  std::cout << "node" << node << " points: " << physPointsCell(0, node,0) << " " << physPointsCell(0, node, 1) <<  std::endl;
      //}
      //for(int i=0; i< numQPsSide; ++i)
      //  std::cout << "qp: " << i << " points: " << refPointsSide(i,0) << " " << refPointsSide(i,1) << " " << refPointsSide(i,2)<< ", name: " << cellType->getName() << std::endl;

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);
      Intrepid2::CellTools<PHX::Device>::setJacobianDet(jacobianSideDet, jacobianSide);

     // std::cout << "Resid: " << __LINE__ << std::endl;
      Intrepid2::CellTools<PHX::Device>::setJacobianInv(invJacobianSide, jacobianSide);
//std::cout << "Resid: " << __LINE__ << std::endl;
      //FST::computeEdgeMeasure(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType, scratch);
      FST::computeCellMeasure(weighted_measure, jacobianSideDet, cubWeightsSide);
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);
//std::cout << "Resid: " << __LINE__ << std::endl;
      intrepidBasis->getValues(basisGrad_refPointsSide, refPointsSide, Intrepid2::OPERATOR_GRAD);
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Transform values of the basis functions
      FST::HGRADtransformVALUE(trans_basis_refPointsSide, basis_refPointsSide);

      FST::HGRADtransformGRAD(trans_gradBasis_refPointsSide, invJacobianSide, basisGrad_refPointsSide);
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Multiply with weighted measure
      //FST::multiplyMeasure(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      dHdt_Nodes = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      forcing_Nodes = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      H_Nodes = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      V_Nodes = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes, numVecFODims);
      gradH_Points = Sacado::createDynRankView(Residual.get_view(), "xxx", numQPsSide, numVecFODims);
      divV_Points = Sacado::createDynRankView(Residual.get_view(), "xxx", numQPsSide);

      for (unsigned int node = 0; node < numSideNodes; ++node){
        dHdt_Nodes(node) = unsteady ? dHdt(elem_LID, node) : ScalarT(Hdiff(elem_LID, node)/ *dt);
        H_Nodes(node) = Hdiff(elem_LID, node) + H0(elem_LID, node);//unsteady ? ScalarT(Hdiff(elem_LID, node) + H0(elem_LID, node)) : ScalarT(H0(elem_LID, node));
        forcing_Nodes(node) = forcing(elem_LID, node);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Nodes(node, dim) = V(elem_LID, node, dim)/1000.0;  //[km/yr]    physPointsCell(0, node, dim)/640.0*(1-dim);//
        }
      }
//std::cout << "Resid: " << __LINE__ << std::endl;
      // This is needed, since evaluate currently sums into
      for (unsigned int qp = 0; qp < numQPsSide; qp++) {
        dHdt_Points(qp) = 0.0;
        H_Points(qp) = 0.0;
        forcing_Points(qp) = 0.0;
        divV_Points(qp) = 0.0;
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Points(qp, dim) = 0.0;
          gradH_Points(qp, dim) = 0.0;
        }
      }
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (unsigned int node = 0; node < numSideNodes; ++node){
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          const MeshScalarT& tmp = trans_basis_refPointsSide(0, node, qp);
          dHdt_Points(qp) += dHdt_Nodes(node) * tmp;
          forcing_Points(qp) += forcing_Nodes(node) * tmp;
          H_Points(qp) += H_Nodes(node) * tmp;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Points(qp, dim) += V_Nodes(node, dim) * tmp;
        }
      }
//std::cout << "Resid: " << __LINE__ << std::endl;
      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (unsigned int node = 0; node < numSideNodes; ++node){
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            const MeshScalarT& tmp = trans_gradBasis_refPointsSide(0, node, qp, dim);
            gradH_Points(qp, dim) += H_Nodes(node) * tmp;
            divV_Points(qp) += V_Nodes(node, dim) * tmp;
          }
        }
      }
//std::cout << "Resid: " << __LINE__ << " " <<numQPsSide << " " <<  numNodes << " " << numVecFODims << " " <<cellDim << " " << weighted_measure(0)  << " " << weighted_measure(1) << " "<< weighted_measure(2)  << " " << weighted_measure(3) << " "<< weighted_measure(4)  << " " << weighted_measure(5) <<  std::endl;
      MeshScalarT h = 0.0;
      for (std::size_t qp = 0; qp < numQPsSide; ++qp) 
        h += weighted_measure(qp);  
    //    std::cout << "Resid: " << __LINE__ << " " << h << std::endl;
      h = sqrt(h);
//std::cout << "Resid: " << __LINE__ << std::endl;
      for (unsigned int node = 0; node < numSideNodes; ++node){
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          ScalarT divHV = divV_Points(qp)* H_Points(qp);
          ScalarT V_norm = 0.0;
          ScalarT V_dot_gradPhi = 0.0;
          ScalarT V_dot_HGrad = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            divHV += gradH_Points(qp, dim)*V_Points(qp,dim);
            V_norm += V_Points(qp, dim)*V_Points(qp, dim);
            V_dot_gradPhi += trans_gradBasis_refPointsSide(0, node, qp, dim)*V_Points(qp, dim);
            V_dot_HGrad += gradH_Points(qp, dim)*V_Points(qp, dim);
          }
          //std::cout << "Resid: " << __LINE__ << std::endl;

          ScalarT tmp = dHdt_Points(qp) + divHV - forcing_Points(qp);

          ScalarT HV_gradPhi = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            HV_gradPhi += H_Points(qp) * V_Points(qp) * trans_gradBasis_refPointsSide(0, node, qp, dim);
          
          res += ((dHdt_Points(qp) - forcing_Points(qp))*trans_basis_refPointsSide(0, node, qp) - HV_gradPhi) * weighted_measure(qp);
          ScalarT invTau = sqrt(4.0/ *dt/ *dt + 4*V_norm*V_norm/h/h + divV_Points(qp)*divV_Points(qp));
          res += tmp * V_dot_gradPhi * weighted_measure(qp) / sqrt(invTau); //SUPG
         // V_norm = sqrt(V_norm+1e-6);
          //res += tmp * h/V_norm*V_dot_gradPhi * weighted_measure(qp); //SUPG

          //res += tmp * (trans_basis_refPointsSide(0, node, qp)+h/V_norm*V_dot_gradPhi) * weighted_measure(qp);
          //res += tmp * trans_basis_refPointsSide(0, node, qp) * weighted_measure(qp)+h/V_norm*V_dot_HGrad*V_dot_gradPhi*weighted_measure(qp);
          
          //res += tmp * trans_basis_refPointsSide(0, node, qp) * weighted_measure(qp);
          ScalarT delta = h*std::min(0.2*V_norm, std::abs(tmp)/std::sqrt(gradH_Points(qp,0)*gradH_Points(qp,0)+gradH_Points(qp,1)*gradH_Points(qp,1)+1e-8));
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) 
            res += delta  *gradH_Points(qp, dim)*trans_gradBasis_refPointsSide(0, node, qp, dim)*weighted_measure(qp);         
        }
        //std::cout << "Resid: " << __LINE__ << std::endl;
        Residual(elem_LID,node) = res;
      }

      // Get the data that corresponds to the side
      auto it = elem_side_map.find(elem_LID);
      if(it == elem_side_map.end())
        continue;   //not on the lateral side

      const int elem_edge = it->second; //Selecting the top edge of the Wedge associated to the edge lateral side
        
      const CellTopologyData_Subcell& edge =  cellType->getCellTopologyData()->edge[elem_edge];
      auto edgeType = Teuchos::rcp(new shards::CellTopology(edge.topology));
      unsigned int numEdgeNodes = edgeType->getNodeCount();
      auto cubatureEdge = cubFactory.create<PHX::Device, RealType, RealType>(*edgeType, cubatureDegree);
      unsigned int edgeDim = edgeType->getDimension();
      unsigned int numQPsEdge = cubatureEdge->getNumPoints();

      // Allocate Temporary Views (should be pre-allocated)
      auto cubPointsEdge = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsEdge, edgeDim);
      auto refPointsEdge = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsEdge, cellDim);
      auto cubWeightsEdge = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsEdge);
      auto basis_refPointsEdge = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsEdge);

      auto jacobianEdge = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim, cellDim);
      auto edge_weighted_measure = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge);
      auto trans_basis_refPointsEdge = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsEdge);
      auto sideNormals = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim);
      auto scratch = Sacado::createDynRankView(jacobianEdge,"XXS", numQPsEdge*cellDim*cellDim);

      auto H_Edge = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);
      //auto V_X_Edge = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);
      auto V_Normal_Edge = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);

      // Pre-Calculate reference element quantities
      cubatureEdge->getCubature(cubPointsEdge, cubWeightsEdge);
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(refPointsEdge, cubPointsEdge, edgeDim, elem_edge, *cellType);
    // std::cout << "Resid: " << __LINE__ << std::endl;

      //for (std::size_t node = 0; node < numNodes; ++node) {
      //  std::cout << "node" << node << " points: " << physPointsCell(0, node,0) << " " << physPointsCell(0, node, 1) << " " << physPointsCell(0, node, 2)<<  std::endl;
      //}
      //for(int i=0; i< numQPsEdge; ++i)
      //  std::cout << "qp: " << i << " points: " << refPointsEdge(i,0) << " " << refPointsEdge(i,1) << " " << refPointsEdge(i,2)<< ", name: " << cellType->getName() << std::endl;

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobianEdge, refPointsEdge, physPointsCell, *cellType);
      Intrepid2::CellTools<PHX::Device>::getPhysicalSideNormals(sideNormals, jacobianEdge, it->second, *cellType );
//std::cout << "Resid: " << __LINE__ << std::endl;
      FST::computeEdgeMeasure(edge_weighted_measure, jacobianEdge, cubWeightsEdge, elem_edge, *cellType, scratch);
      //std::cout << "Resid: " << __LINE__ << std::endl;
      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsEdge, refPointsEdge, Intrepid2::OPERATOR_VALUE);
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Transform values of the basis functions
      FST::HGRADtransformVALUE(trans_basis_refPointsEdge, basis_refPointsEdge);
      
      // This is needed, since evaluate currently sums into
      for (unsigned int qp = 0; qp < numQPsEdge; qp++) {
        H_Edge(qp) = 0.0;
        V_Normal_Edge(qp) = 0.0;
        //V_X_Edge(qp) = 0.0;
        MeshScalarT norm = 0.0;
        for (std::size_t dim = 0; dim < numVecFODims; ++dim)
          norm += sideNormals(0, qp, dim)*sideNormals(0, qp, dim);
        norm = std::sqrt(norm);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim)
          sideNormals(0, qp, dim) /= norm;          
      }
//std::cout << "Resid: " << __LINE__ << std::endl;
      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (unsigned int i = 0; i < numEdgeNodes; ++i){
        std::size_t node = edge.node[i];
        for (std::size_t qp = 0; qp < numQPsEdge; ++qp) {
          const MeshScalarT& edge_basis = trans_basis_refPointsEdge(0, node, qp);
          H_Edge(qp) += H_Nodes(node) * edge_basis;
          //V_X_Edge(qp) += V_Nodes(node,0) * tmp;
          auto normal_norm = 0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Normal_Edge(qp) += V_Nodes(node, dim) * edge_basis * sideNormals(0, qp, dim);
        }
      }
//std::cout << "Resid: " << __LINE__ << std::endl;
     // for (unsigned int qp = 0; qp < numQPsEdge; qp++)
      //  std::cout << "qp: " << qp << ", Normal V: " << V_Normal_Edge(qp) << ", N:" << sideNormals(0, qp, 0) << ", " <<  sideNormals(0, qp, 1) <<  ", H: " << H_Edge(qp) << ", measure:  " <<  edge_weighted_measure(qp) << std::endl;

      for (unsigned int i = 0; i < numEdgeNodes; ++i){
        std::size_t node = edge.node[i];
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsEdge; ++qp) { 
          if(V_Normal_Edge(qp) > 0)   
            res += H_Edge(qp) * V_Normal_Edge(qp) * trans_basis_refPointsEdge(0, node, qp) * edge_weighted_measure(qp);
        }
        Residual(elem_LID,node) += res;
      }
    }
  }
}

} // namespace LandIce
