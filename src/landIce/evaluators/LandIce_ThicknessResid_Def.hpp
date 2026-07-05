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
#include "Kokkos_ViewFactory.hpp"

#include "Albany_MeshSpecs.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "LandIce_ThicknessResid.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
ThicknessResid<EvalT, Traits>::
ThicknessResid(const Teuchos::ParameterList& p,
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

  if(p.isParameter("SMB Name")) {
   SMB = decltype(SMB)(p.get<std::string> ("SMB Name"), dl->node_scalar);
   have_SMB = true;
   this->addDependentField(SMB);
  } else {
    have_SMB = false;
  }

  this->addEvaluatedField(Residual);

  dt = p.get<Teuchos::RCP<double> >("Time Step Ptr");

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = p.get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");

  sideSetName  = p.get<std::string> ("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  auto av_v_layout = dl_side->node_vector;
  V = decltype(V)(p.get<std::string>("Averaged Velocity Variable Name"), av_v_layout);
  this->addDependentField(V);

  std::cout << "\nSideSetName: " << sideSetName << std::endl;
  for (const auto &ele : dl->side_layouts) 
    std::cout <<"Layout: " <<ele.first << std::endl;

  lateralSideSetName = p.isParameter("Lateral Side Set Name") ? p.get<std::string>("Lateral Side Set Name") : std::string("lateralside");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(lateralSideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Lateral side data layout not found.\n");

  this->setName("ThicknessResid"+PHX::print<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  numVecFODims  = std::min(dims[2], PHX::DataLayout::size_type(2));

  dl->qp_gradient->dimensions(dims);
  cellDim = dims[2];

  const CellTopologyData * const elem_top = &meshSpecs->ctd;
  TEUCHOS_TEST_FOR_EXCEPTION (elem_top->dimension != 3, std::runtime_error,
                              "Error! This evaluator expects a 3D cell.\n");

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
void ThicknessResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{
  physPointsCell = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, cellDim);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThicknessResid<EvalT, Traits>::
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

  Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  if (it_ss != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it_ss->second;

    Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basisGrad_refPointsSide;

    Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> invJacobianSide;
    //Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_det;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_gradBasis_refPointsSide;
    //Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> scratch;

   Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> SMB_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> H_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Side;

    Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> SMB_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> H_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> gradH_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> divV_Side;

    // Loop over the sides that form the boundary condition
    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].ws_elem_idx;
      const int elem_side = sideSet[iSide].side_pos;
      const CellTopologyData_Subcell& side =  cellType->getCellTopologyData()->side[elem_side];
      sideType = Teuchos::rcp(new shards::CellTopology(side.topology));
      unsigned int numSideNodes = sideType->getNodeCount();
      Intrepid2::DefaultCubatureFactory cubFactory;
      cubatureSide = cubFactory.create<PHX::Device, RealType, RealType>(*sideType, cubatureDegree);
      unsigned int sideDims = sideType->getDimension();
      unsigned int numQPsSide = cubatureSide->getNumPoints();

      // Allocate Temporary Views (should be pre-allocated)
      cubPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, sideDims);
      refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, cellDim);
      cubWeightsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide);
      basis_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide);
      basisGrad_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide, cellDim);

      jacobianSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      invJacobianSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      weighted_measure = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      trans_basis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      trans_gradBasis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide, cellDim);
      //weighted_trans_basis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      scratch = Kokkos::createDynRankView(jacobianSide,"XXS", numQPsSide*cellDim*cellDim);

      dHdt_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      SMB_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      H_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      V_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide, numVecFODims);

      // Pre-Calculate reference element quantities
      cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

      // Copy the coordinate data over to a temp container
     for (std::size_t node = 0; node < numNodes; ++node) {
       for (std::size_t dim = 0; dim < cellDim; ++dim)
         physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
         physPointsCell(0, node, cellDim-1) = -1.0; //set z=-1 on internal cell nodes and z=0 side (see next lines).
     }
     for (unsigned int i = 0; i < numSideNodes; ++i)
       physPointsCell(0, side.node[i], cellDim-1) = 1.0;  //set z=0 on side

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      //for (std::size_t node = 0; node < numNodes; ++node) {
      //  std::cout << "node" << node << " points: " << physPointsCell(0, node,0) << " " << physPointsCell(0, node, 1) << " " << physPointsCell(0, node, 2)<<  std::endl;
      //}
      //for(int i=0; i< numQPsSide; ++i)
      //  std::cout << "qp: " << i << " points: " << refPointsSide(i,0) << " " << refPointsSide(i,1) << " " << refPointsSide(i,2)<< ", name: " << cellType->getName() << std::endl;

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid2::CellTools<PHX::Device>::setJacobianInv(invJacobianSide, jacobianSide);

      //FST::computeEdgeMeasure(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType, scratch);
      FST::computeFaceMeasure(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType, scratch);

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      intrepidBasis->getValues(basisGrad_refPointsSide, refPointsSide, Intrepid2::OPERATOR_GRAD);

      // Transform values of the basis functions
      FST::HGRADtransformVALUE(trans_basis_refPointsSide, basis_refPointsSide);

      FST::HGRADtransformGRAD(trans_gradBasis_refPointsSide, invJacobianSide, basisGrad_refPointsSide);

      // Multiply with weighted measure
      //FST::multiplyMeasure(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      dHdt_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      SMB_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      H_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      V_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes, numVecFODims);
      gradH_Side = createDynRankView(Residual.get_view(), "xxx", numQPsSide, numVecFODims);
      divV_Side = createDynRankView(Residual.get_view(), "xxx", numQPsSide);

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        dHdt_Cell(node) = unsteady ? dHdt(elem_LID, node) : ScalarT(Hdiff(elem_LID, node)/ *dt);
        H_Cell(node) = Hdiff(elem_LID, node) + H0(elem_LID, node);//unsteady ? ScalarT(Hdiff(elem_LID, node) + H0(elem_LID, node)) : ScalarT(H0(elem_LID, node));
        SMB_Cell(node) = have_SMB ? SMB(elem_LID, node) : ScalarT(0.0);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Cell(node, dim) = V(iSide, i, dim)/1000.0;  //[km/yr]
        }
      }

      // This is needed, since evaluate currently sums into
      for (unsigned int qp = 0; qp < numQPsSide; qp++) {
        dHdt_Side(qp) = 0.0;
        H_Side(qp) = 0.0;
        SMB_Side(qp) = 0.0;
        divV_Side(qp) = 0.0;
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Side(qp, dim) = 0.0;
          gradH_Side(qp, dim) = 0.0;
        }
      }

      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          const MeshScalarT& tmp = trans_basis_refPointsSide(0, node, qp);
          dHdt_Side(qp) += dHdt_Cell(node) * tmp;
          SMB_Side(qp) += SMB_Cell(node) * tmp;
          H_Side(qp) += H_Cell(node) * tmp;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Side(qp, dim) += V_Cell(node, dim) * tmp;
        }
      }

      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (unsigned int i = 0; i < numSideNodes; ++i){
          std::size_t node = side.node[i];
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            const MeshScalarT& tmp = trans_gradBasis_refPointsSide(0, node, qp, dim);
            gradH_Side(qp, dim) += H_Cell(node) * tmp;
            divV_Side(qp) += V_Cell(node, dim) * tmp;
          }
        }
      }

      MeshScalarT h = 0.0;
      for (std::size_t qp = 0; qp < numQPsSide; ++qp) 
        h += weighted_measure(qp);  
      h = sqrt(h);

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          ScalarT divHV = divV_Side(qp)* H_Side(qp);
          ScalarT V_norm = 0.0;
          ScalarT V_dot_gradPhi = 0.0;
          ScalarT V_dot_HGrad = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            divHV += gradH_Side(qp, dim)*V_Side(qp,dim);
            V_norm += V_Side(qp, dim)*V_Side(qp, dim);
            V_dot_gradPhi += trans_gradBasis_refPointsSide(0, node, qp, dim)*V_Side(qp, dim);
            V_dot_HGrad += gradH_Side(qp, dim)*V_Side(qp, dim);
          }
          V_norm = sqrt(V_norm+1e-6);

          ScalarT tmp = dHdt_Side(qp) + divHV - 3e-4;// - SMB_Side(qp);

          ScalarT HV_gradPhi = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            HV_gradPhi += H_Side(qp) * V_Side(qp) * trans_gradBasis_refPointsSide(0, node, qp, dim);
          
          res += ((dHdt_Side(qp) - 3e-4)*trans_basis_refPointsSide(0, node, qp) - HV_gradPhi) * weighted_measure(qp);
          res += tmp * h/V_norm*V_dot_gradPhi * weighted_measure(qp); //SUPG

          //res += tmp * (trans_basis_refPointsSide(0, node, qp)+h/V_norm*V_dot_gradPhi) * weighted_measure(qp);
          //res += tmp * trans_basis_refPointsSide(0, node, qp) * weighted_measure(qp)+h/V_norm*V_dot_HGrad*V_dot_gradPhi*weighted_measure(qp);
          
          //res += tmp * trans_basis_refPointsSide(0, node, qp) * weighted_measure(qp);
          //for (std::size_t dim = 0; dim < numVecFODims; ++dim) 
           // res += 3*h*V_norm/2.0*gradH_Side(qp, dim)*trans_gradBasis_refPointsSide(0, node, qp, dim)*weighted_measure(qp);         
        }
        Residual(elem_LID,node) = res;
      }

      // Get the data that corresponds to the side
      auto it = elem_side_map.find(elem_LID);
      if(it == elem_side_map.end())
        continue;   //not on the lateral side

      const int elem_edge = it->second+3; //hack. Selecting the top edge of the Wedge associated to the edge lateral side
        
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

      auto jacobianEdge = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim, cellDim);
      auto edge_weighted_measure = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge);
      auto trans_basis_refPointsEdge = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsEdge);
      auto sideNormals = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim);
      auto scratch = Kokkos::createDynRankView(jacobianEdge,"XXS", numQPsEdge*cellDim*cellDim);

      auto H_Edge = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);
      //auto V_X_Edge = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);
      auto V_Normal_Edge = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsEdge);

      // Pre-Calculate reference element quantities
      cubatureEdge->getCubature(cubPointsEdge, cubWeightsEdge);

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(refPointsEdge, cubPointsEdge, edgeDim, elem_edge, *cellType);

      

      //for (std::size_t node = 0; node < numNodes; ++node) {
      //  std::cout << "node" << node << " points: " << physPointsCell(0, node,0) << " " << physPointsCell(0, node, 1) << " " << physPointsCell(0, node, 2)<<  std::endl;
      //}
      //for(int i=0; i< numQPsEdge; ++i)
      //  std::cout << "qp: " << i << " points: " << refPointsEdge(i,0) << " " << refPointsEdge(i,1) << " " << refPointsEdge(i,2)<< ", name: " << cellType->getName() << std::endl;

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobianEdge, refPointsEdge, physPointsCell, *cellType);
      Intrepid2::CellTools<PHX::Device>::getPhysicalSideNormals(sideNormals, jacobianEdge, it->second, *cellType );

      FST::computeEdgeMeasure(edge_weighted_measure, jacobianEdge, cubWeightsEdge, elem_edge, *cellType, scratch);
      
      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsEdge, refPointsEdge, Intrepid2::OPERATOR_VALUE);

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

      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (unsigned int i = 0; i < numEdgeNodes; ++i){
        std::size_t node = edge.node[i];
        for (std::size_t qp = 0; qp < numQPsEdge; ++qp) {
          const MeshScalarT& tmp = trans_basis_refPointsEdge(0, node, qp);
          H_Edge(qp) += H_Cell(node) * tmp;
          //V_X_Edge(qp) += V_Cell(node,0) * tmp;
          auto normal_norm = 0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Normal_Edge(qp) += V_Cell(node, dim) * tmp * sideNormals(0, qp, dim);
        }
      }

      //for (unsigned int qp = 0; qp < numQPsEdge; qp++)
      //  std::cout << "qp: " << qp << ", Normal V: " << V_Normal_Edge(qp) << " | " <<V_X_Edge(qp)<< ", N:" << sideNormals(0, qp, 0) << ", " <<  sideNormals(0, qp, 1) <<  ", H: " << H_Edge(qp) << ", measure:  " <<  edge_weighted_measure(qp) << std::endl;

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
