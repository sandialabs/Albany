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
  dH       (p.get<std::string> ("Thickness Increment Variable Name"), dl->node_scalar),
  H0       (p.get<std::string> ("Past Thickness Name"), dl->node_scalar),
  coordVec (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector),
  Residual (p.get<std::string> ("Residual Name"), dl->node_scalar)
{
  this->addDependentField(dH);
  this->addDependentField(H0);
  this->addDependentField(coordVec);
  if(p.isParameter("SMB Name")) {
   SMB = decltype(SMB)(p.get<std::string> ("SMB Name"), dl->node_scalar);
   have_SMB = true;
   this->addDependentField(SMB);
  } else {
    have_SMB = false;
  }

  this->addEvaluatedField(Residual);

  dt = p.get<Teuchos::RCP<double> >("Time Step Ptr");

  Teuchos::RCP<const Albany::MeshSpecs> meshSpecs = p.get<Teuchos::RCP<const Albany::MeshSpecs> >("Mesh Specs Struct");

  sideSetName  = p.get<std::string> ("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  auto av_v_layout = dl_side->node_vector;
  V = decltype(V)(p.get<std::string>("Averaged Velocity Variable Name"), av_v_layout);
  this->addDependentField(V);

  this->setName("ThicknessResid"+PHX::print<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  numVecFODims  = std::min(dims[2], PHX::DataLayout::size_type(2));

  dl->qp_gradient->dimensions(dims);
  cellDims = dims[2];

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

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
  physPointsCell = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, cellDims);
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
    Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_det;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_gradBasis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_trans_basis_refPointsSide;
    Kokkos::DynRankView<MeshScalarT, PHX::Device> scratch;

    Kokkos::DynRankView<ScalarT, PHX::Device> dofSide;
    Kokkos::DynRankView<ScalarT, PHX::Device> dofSideVec;
    Kokkos::DynRankView<ScalarT, PHX::Device> dH_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> SMB_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> H0_Side;
    Kokkos::DynRankView<ScalarT, PHX::Device> V_Side;

    Kokkos::DynRankView<ScalarT, PHX::Device> dH_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> SMB_Cell;
    Kokkos::DynRankView<ScalarT, PHX::Device> H0_Cell;
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

      // Allocate Temporary Views
      cubPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, sideDims);
      refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide, cellDims);
      cubWeightsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPsSide);
      basis_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide);
      basisGrad_refPointsSide = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPsSide, cellDims);

      jacobianSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDims, cellDims);
      invJacobianSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide, cellDims, cellDims);
      jacobianSide_det = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      weighted_measure = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numQPsSide);
      trans_basis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      trans_gradBasis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide, cellDims);
      weighted_trans_basis_refPointsSide = Kokkos::createDynRankView(coordVec.get_view(), "XXX", 1, numNodes, numQPsSide);
      scratch = Kokkos::createDynRankView(jacobianSide,"XXS", numQPsSide*cellDims*cellDims);

      dofSide = Kokkos::createDynRankView(Residual.get_view(), "XXX", 1, numQPsSide);
      dofSideVec = Kokkos::createDynRankView(Residual.get_view(), "XXX", 1, numQPsSide, numVecFODims);
      dH_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      SMB_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      H0_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide);
      V_Side = Kokkos::createDynRankView(Residual.get_view(), "XXX", numQPsSide, numVecFODims);

      // Pre-Calculate reference element quantities
      cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

      // Copy the coordinate data over to a temp container
     for (std::size_t node = 0; node < numNodes; ++node) {
       for (std::size_t dim = 0; dim < cellDims; ++dim)
         physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
       physPointsCell(0, node, cellDims-1) = -1.0; //set z=-1 on internal cell nodes and z=0 side (see next lines).
     }
     for (unsigned int i = 0; i < numSideNodes; ++i)
       physPointsCell(0, side.node[i], cellDims-1) = 0.0;  //set z=0 on side

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid2::CellTools<PHX::Device>::setJacobianInv(invJacobianSide, jacobianSide);

      Intrepid2::CellTools<PHX::Device>::setJacobianDet(jacobianSide_det, jacobianSide);

      if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
        FST::computeEdgeMeasure(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType, scratch);
      } else { //for 3D, get weighted face measure
        FST::computeFaceMeasure(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType, scratch);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      intrepidBasis->getValues(basisGrad_refPointsSide, refPointsSide, Intrepid2::OPERATOR_GRAD);

      // Transform values of the basis functions
      FST::HGRADtransformVALUE(trans_basis_refPointsSide, basis_refPointsSide);

      FST::HGRADtransformGRAD(trans_gradBasis_refPointsSide, invJacobianSide, basisGrad_refPointsSide);

      // Multiply with weighted measure
      FST::multiplyMeasure(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      dH_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      SMB_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      H0_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes);
      V_Cell = createDynRankView(Residual.get_view(), "xxx", numNodes, numVecFODims);
      gradH_Side = createDynRankView(Residual.get_view(), "xxx", numQPsSide, numVecFODims);
      divV_Side = createDynRankView(Residual.get_view(), "xxx", numQPsSide);

      std::map<LO, std::size_t>::const_iterator it;

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        dH_Cell(node) = dH(elem_LID, node);
        H0_Cell(node) = H0(elem_LID, node);
        SMB_Cell(node) = have_SMB ? SMB(elem_LID, node) : ScalarT(0.0);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
          V_Cell(node, dim) = V(iSide, i, dim);
        }
      }

      // This is needed, since evaluate currently sums into
      for (unsigned int qp = 0; qp < numQPsSide; qp++) {
        dH_Side(qp) = 0.0;
        H0_Side(qp) = 0.0;
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
          dH_Side(qp) += dH_Cell(node) * tmp;
          SMB_Side(qp) += SMB_Cell(node) * tmp;
          H0_Side(qp) += H0_Cell(node) * tmp;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Side(qp, dim) += V_Cell(node, dim) * tmp;
        }
      }

      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (unsigned int i = 0; i < numSideNodes; ++i){
          std::size_t node = side.node[i];
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            const MeshScalarT& tmp = trans_gradBasis_refPointsSide(0, node, qp, dim);
            gradH_Side(qp, dim) += H0_Cell(node) * tmp;
            divV_Side(qp) += V_Cell(node, dim) * tmp;
          }
        }
      }

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          ScalarT divHV = divV_Side(qp)* H0_Side(qp);
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            divHV += gradH_Side(qp, dim)*V_Side(qp,dim);

         ScalarT tmp = dH_Side(qp) + (*dt/1000.0) * divHV - *dt*SMB_Side(qp);
          res +=tmp * weighted_trans_basis_refPointsSide(0, node, qp);
        }
        Residual(elem_LID,node) = res;
      }
    }
  }
}

} // namespace LandIce
