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

  if(p.isParameter("Forcing Name")) {
   forcing = decltype(forcing)(p.get<std::string> ("Forcing Name"), dl->node_scalar);
   have_forcing = true;
   this->addDependentField(forcing);
  } else {
    have_forcing = false;
  }

  supg = graph_viscosity = edge_stabilization =false;
  if(p.isParameter("Stabilization")) {
    const auto stabilization = p.get<std::string>("Stabilization");
    if(stabilization == "SUPG")
      supg = true;
    else if (stabilization == "Graph Viscosity")
      graph_viscosity = true;
    else if (stabilization == "Edge Stabilization")
      edge_stabilization = true;
    else 
      TEUCHOS_TEST_FOR_EXCEPTION (stabilization != "None", std::runtime_error,
        "Error! Stabilization \"" << stabilization << "\" not supported.\nSupported stabilizations are: \"SUPG\", \"Graph Viscosity\", \"Edge Stabilization\" and \"None\".\n");
  }
  
  lump_mass = p.isParameter("Lump Mass Matrix") ? p.get<bool>("Lump Mass Matrix") : false;

  this->addEvaluatedField(Residual);

  //only used in the steady case
  if (!unsteady)
    dt = p.get<Teuchos::RCP<double> >("Time Step Ptr");
  else
    dt = Teuchos::rcp(new double(-1.0));
  

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
  physPointsCell = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, cellDim);
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
  physPointsCell = Intrepid2::Impl::createMatchingDynRankView(coordVec.get_view(), "XXX", 1, numNodes, cellDim);

  if (it_ss != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it_ss->second;

    Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide;
    Kokkos::DynRankView<RealType, PHX::Device> basisGrad_refPointsSide;

    Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> invJacobianSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_gradBasis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> scratch;

    Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> forcing_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> H_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Side;

    Kokkos::DynRankView<ScalarT, PHX::Device> dHdt_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> forcing_Cell;
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
      TEUCHOS_TEST_FOR_EXCEPTION(numSideNodes != 3, std::runtime_error, "Mass Lumping and edge stabilization currently assumes P1 triangles.\n");

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

      jacobianSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      invJacobianSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDim, cellDim);
      weighted_measure = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      trans_basis_refPointsSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      trans_gradBasis_refPointsSide = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide, cellDim);
      scratch = Sacado::createDynRankView(jacobianSide,"XXS", numQPsSide*cellDim);

      dHdt_Side = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      forcing_Side = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      H_Side = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      V_Side = Sacado::createDynRankView(Residual.get_view(), "XXX", numQPsSide, numVecFODims);

      // Pre-Calculate reference element quantities
      cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

      // Copy the coordinate data over to a temp container
      for (std::size_t node = 0; node < numNodes; ++node) {
        for (std::size_t dim = 0; dim < cellDim-1; ++dim)
          physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
        physPointsCell(0, node, cellDim-1) = -1.0; //set z=-1 on internal cell nodes and z=0 side (see next lines).
      }
      for (unsigned int i = 0; i < numSideNodes; ++i)
        physPointsCell(0, side.node[i], cellDim-1) = 1.0;  //set z=1 on side

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
      dHdt_Cell = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      forcing_Cell = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      H_Cell = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes);
      V_Cell = Sacado::createDynRankView(Residual.get_view(), "xxx", numNodes, numVecFODims);
      gradH_Side = Sacado::createDynRankView(Residual.get_view(), "xxx", numQPsSide, numVecFODims);
      divV_Side = Sacado::createDynRankView(Residual.get_view(), "xxx", numQPsSide);

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        dHdt_Cell(node) = unsteady ? dHdt(elem_LID, node) : ScalarT(Hdiff(elem_LID, node)/ *dt);
        H_Cell(node) = Hdiff(elem_LID, node) + H0(elem_LID, node);//unsteady ? ScalarT(Hdiff(elem_LID, node) + H0(elem_LID, node)) : ScalarT(H0(elem_LID, node));
        forcing_Cell(node) = have_forcing ? forcing(elem_LID, node) : ScalarT(0.0);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Cell(node, dim) = V(iSide, i, dim)/1000.0;  //[km/yr]     physPointsCell(0, node, dim)/640.0*(1-dim);//
        }
      }

      // This is needed, since evaluate currently sums into
      for (unsigned int qp = 0; qp < numQPsSide; qp++) {
        dHdt_Side(qp) = 0.0;
        H_Side(qp) = 0.0;
        forcing_Side(qp) = 0.0;
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
          forcing_Side(qp) += forcing_Cell(node) * tmp;
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

      MeshScalarT area = 0.0;
      for (std::size_t qp = 0; qp < numQPsSide; ++qp) 
        area += weighted_measure(0,qp);  
      MeshScalarT h = sqrt(area);

      
      RealType Q[3][3] = {{0.0}};

      if(graph_viscosity) { 
        RealType A[3][3] = {{0.0}};   
        // Element advection matrix
        for (int i = 0; i < 3; ++i) {
          std::size_t node_i = side.node[i];
          for (int j = 0; j < 3; ++j) {
            std::size_t node_j = side.node[j];
            for (int qp = 0; qp < numQPsSide; ++qp) {

              ScalarT V_dot_gradN_i = 0.0;
              for (int dim = 0; dim < 2; ++dim)
                V_dot_gradN_i += V_Side(qp,dim) * trans_gradBasis_refPointsSide(0,node_i,qp,dim);
              ScalarT Aij = trans_basis_refPointsSide(0,node_j,qp) * V_dot_gradN_i * weighted_measure(0, qp);
              A[i][j] -= Albany::convertScalar<RealType>(Aij); //discard derivatives of stabilization term
            }
          }
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = i+1; j < 3; ++j) {

            RealType nu_ij = std::max(RealType(0.0), std::max(A[i][j], A[j][i]));

            Q[i][j] = -nu_ij;
            Q[j][i] = -nu_ij;

            Q[i][i] += nu_ij;
            Q[j][j] += nu_ij;
          }
        }
      }

      if(edge_stabilization) {
        // -----------------------------------------------------------------------------
        // Parameter-free symmetric edge stabilization
        //
        // Q(H,phi) = int_K Theta_tilde(H) . Theta_tilde(phi) dx
        //
        // Theta_tilde(H) = sum_e sqrt(theta_e) (g_e H) W_e
        //
        // For pure advection:
        //   theta_e = 0.5 * h_e * |Vbar_e . t_e|
        //
        // Lowest-order Nedelec edge basis:
        //   W_ij = N_i grad(N_j) - N_j grad(N_i)
        //
        // Edge orientation is arbitrary but must be used consistently.
        // -----------------------------------------------------------------------------



        // Local oriented edges of the triangle.
        // The orientation itself does not matter.
        constexpr int edgeNodes[3][2] = {
            {0, 1},
            {1, 2},
            {2, 0}
        };

        // Compute theta_e for each edge.
        ScalarT theta[3];

        for (int e = 0; e < 3; ++e) {

          const int nodei = side.node[edgeNodes[e][0]];
          const int nodej = side.node[edgeNodes[e][1]];

          MeshScalarT edge_vec[2];
          MeshScalarT h_e2 = 0.0;

          for (int dim = 0; dim < 2; ++dim) {
            edge_vec[dim] =
                coordVec(elem_LID,nodej,dim)
              - coordVec(elem_LID,nodei,dim);

            h_e2 += edge_vec[dim] * edge_vec[dim];
          }

          const MeshScalarT h_e = std::sqrt(h_e2);

          // Mean tangential velocity along the edge.
          //
          // Since V is P1, the edge average is exactly the average
          // of the endpoint values.
          ScalarT ubar_e = 0.0;

          for (int dim = 0; dim < 2; ++dim) {

            const MeshScalarT t_e = edge_vec[dim] / h_e;

            ubar_e +=
                0.5
              * (V_Cell(nodei,dim) + V_Cell(nodej,dim))
              * t_e;
          }

          // Pure-advection limit of the paper's stabilization.
          theta[e] = 0.5 * h_e * std::abs(ubar_e);
        }


        // Build the local 3x3 stabilization matrix
        //
        //   Q_K(i,j)
        //     = int_K Theta_tilde(N_j) . Theta_tilde(N_i) dx
        //
        // Algebraically this is
        //
        //   Q_K = G^T sqrt(Theta) M sqrt(Theta) G.
        //
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            Q[i][j] = 0.0;


        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {

          // thetaN[a][dim] = Theta_tilde(N_a)
          ScalarT thetaN[3][2];

          for (int a = 0; a < 3; ++a)
            for (int dim = 0; dim < 2; ++dim)
              thetaN[a][dim] = 0.0;


          for (int e = 0; e < 3; ++e) {

            const int i = edgeNodes[e][0];
            const int j = edgeNodes[e][1];

            const ScalarT sqrt_theta = std::sqrt(theta[e]+1e-12);

            for (int dim = 0; dim < 2; ++dim) {

              // Lowest-order Nedelec edge basis
              //
              // W_ij = N_i grad(N_j) - N_j grad(N_i)
              const MeshScalarT W_e =
                  trans_basis_refPointsSide(0,side.node[i],qp)
                * trans_gradBasis_refPointsSide(0,side.node[j],qp,dim)
                -
                  trans_basis_refPointsSide(0,side.node[j],qp)
                * trans_gradBasis_refPointsSide(0,side.node[i],qp,dim);

              // g_e,i = -1
              // g_e,j = +1
              thetaN[i][dim] -= sqrt_theta * W_e;
              thetaN[j][dim] += sqrt_theta * W_e;
            }
          }


          // Q_ij = int Theta_tilde(N_i) . Theta_tilde(N_j)
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {

              ScalarT qij = 0.0;

              for (int dim = 0; dim < 2; ++dim)
                qij += thetaN[i][dim] * thetaN[j][dim];

              ScalarT Qij = qij * weighted_measure(qp);
              Q[i][j] += Albany::convertScalar<RealType>(Qij); //discard derivatives of stabilization term
            }
          }
        }
      }

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        ScalarT res = 0;
        if(lump_mass)
          res += (dHdt_Cell(node) - forcing_Cell(node))*area/3.0;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          ScalarT divHV = divV_Side(qp)* H_Side(qp);
          ScalarT V_norm2 = 0.0;
          ScalarT V_dot_gradPhi = 0.0;
          ScalarT V_dot_HGrad = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            divHV += gradH_Side(qp, dim)*V_Side(qp,dim);
            V_norm2 += V_Side(qp, dim)*V_Side(qp, dim);
            V_dot_gradPhi += trans_gradBasis_refPointsSide(0, node, qp, dim)*V_Side(qp, dim);
            V_dot_HGrad += gradH_Side(qp, dim)*V_Side(qp, dim);
          }          

          ScalarT advective_rate = 0.0;  // 2|V|/ h_V,  h_V = 2|V|/sum(|V \nabla Phi|)
          for (unsigned int i = 0; i < numSideNodes; ++i){
            ScalarT V_dot_gradN = 0.0;

            for (std::size_t dim = 0; dim < numVecFODims; ++dim)
              V_dot_gradN += V_Side(qp,dim) * trans_gradBasis_refPointsSide(0,side.node[i],qp,dim);

            advective_rate += std::abs(V_dot_gradN);
          }

          
          ScalarT HV_gradPhi = 0.0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            HV_gradPhi += H_Side(qp) * V_Side(qp, dim) * trans_gradBasis_refPointsSide(0, node, qp, dim);
          
          if(!lump_mass)
            res += (dHdt_Side(qp) - forcing_Side(qp))*trans_basis_refPointsSide(0, node, qp) * weighted_measure(0, qp);
          res -=  HV_gradPhi * weighted_measure(0, qp);
          //std::cout << "Time Step: " << workset.time_step << std::endl;
          
          if(supg) {
            ScalarT tmp = dHdt_Side(qp) + divHV - forcing_Side(qp);
            if(workset.time_step != 0){
              ScalarT invTau = sqrt(4.0/ workset.time_step / workset.time_step + advective_rate*advective_rate + divV_Side(qp)*divV_Side(qp));
              res += tmp * V_dot_gradPhi * weighted_measure(0, qp) / invTau; //SUPG
            }
            ScalarT delta = V_norm2/(advective_rate+1e-12);
            //ScalarT delta = h*std::min(0.2*sqrt(V_norm2+1e-12), std::abs(tmp)/std::sqrt(gradH_Side(qp,0)*gradH_Side(qp,0)+gradH_Side(qp,1)*gradH_Side(qp,1)+1e-12));
            for (std::size_t dim = 0; dim < numVecFODims; ++dim) 
              res += delta  *gradH_Side(qp, dim)*trans_gradBasis_refPointsSide(0, node, qp, dim)*weighted_measure(0, qp); 
          }       
        }

        if(graph_viscosity || edge_stabilization) {          
          for (int j = 0; j < 3; ++j)
            res += Q[i][j] * H_Cell(side.node[j]);
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

      auto jacobianEdge = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim, cellDim);
      auto edge_weighted_measure = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge);
      auto trans_basis_refPointsEdge = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsEdge);
      auto sideNormals = Sacado::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsEdge, cellDim);
      auto sideScratch = Sacado::createDynRankView(jacobianEdge,"XXS", numQPsEdge*cellDim);

      auto H_Edge = Intrepid2::Impl::createMatchingDynRankView(Residual.get_view(), "XXX", numQPsEdge);
      auto V_Normal_Edge = Intrepid2::Impl::createMatchingDynRankView(Residual.get_view(), "XXX", numQPsEdge);

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

      FST::computeEdgeMeasure(edge_weighted_measure, jacobianEdge, cubWeightsEdge, elem_edge, *cellType, sideScratch);
      
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
          const MeshScalarT& edge_basis = trans_basis_refPointsEdge(0, node, qp);
          H_Edge(qp) += H_Cell(node) * edge_basis;
          //V_X_Edge(qp) += V_Cell(node,0) * tmp;
          auto normal_norm = 0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Normal_Edge(qp) += V_Cell(node, dim) * edge_basis * sideNormals(0, qp, dim);
        }
      }

     // for (unsigned int qp = 0; qp < numQPsEdge; qp++)
      //  std::cout << "qp: " << qp << ", Normal V: " << V_Normal_Edge(qp) << ", N:" << sideNormals(0, qp, 0) << ", " <<  sideNormals(0, qp, 1) <<  ", H: " << H_Edge(qp) << ", measure:  " <<  edge_weighted_measure(0, qp) << std::endl;

      for (unsigned int i = 0; i < numEdgeNodes; ++i){
        std::size_t node = edge.node[i];
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsEdge; ++qp) { 
          if(V_Normal_Edge(qp) > 0)
            if(graph_viscosity)   
              res += H_Cell(node) * V_Normal_Edge(qp) * trans_basis_refPointsEdge(0, node, qp) * edge_weighted_measure(0, qp);
            else 
              res += H_Edge(qp) * V_Normal_Edge(qp) * trans_basis_refPointsEdge(0, node, qp) * edge_weighted_measure(0, qp);
        }
        Residual(elem_LID,node) += res;
      }
    }
  }
}

} // namespace LandIce
