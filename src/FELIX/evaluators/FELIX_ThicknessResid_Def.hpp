//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
ThicknessResid<EvalT, Traits>::
ThicknessResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  H        (p.get<std::string> ("Thickness Variable Name"), dl->node_scalar),
  H0       (p.get<std::string> ("Old Thickness Name"), dl->node_scalar),
  V        (p.get<std::string> ("Averaged Velocity Variable Name"), dl->node_vector),
  coordVec ("Coord Vec", dl->vertices_vector),
  Residual (p.get<std::string> ("Residual Name"), dl->node_scalar)
{

  dt = p.get<double>("Time Step");

  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = p.get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");

  this->addDependentField(H);
  this->addDependentField(H0);
  this->addDependentField(V);
  this->addDependentField(coordVec);

  this->addEvaluatedField(Residual);


  this->setName("ThicknessResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  numVecDims  = std::min((size_t)dims[2], (size_t)2);

  dl->qp_gradient->dimensions(dims);
  numDims = cellDims = dims[2];



  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepidBasis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology(elem_top));

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  cubatureCell = cubFactory.create(*cellType, 1); //meshSpecs->cubatureDegree);
  cubatureDegree = plist->isParameter("Cubature Degree") ? plist->get<int>("Cubature Degree") : meshSpecs->cubatureDegree;
  numNodes = intrepidBasis->getCardinality();

  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  dofCellVec.resize(1, numNodes, numVecDims);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
#ifdef OUTPUT_TO_SCREEN
*out << " in FELIX Thickness residual! " << std::endl;
*out << " numDims = " << numDims << std::endl;
*out << " numNodes = " << numNodes << std::endl; 
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThicknessResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(H0,fm);
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(coordVec, fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThicknessResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST; 

  // Initialize residual to 0.0
  Kokkos::deep_copy(Residual.get_kokkos_view(), ScalarT(0.0));

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find("upperside");



  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

    int numLayers = layeredMeshNumbering.numLayers;

    Intrepid::FieldContainer<ScalarT> H_Side;
    Intrepid::FieldContainer<ScalarT> H0_Side;
    Intrepid::FieldContainer<ScalarT> V_Side;

    std::map<LO, std::size_t> baseIds;


    // Loop over the sides that form the boundary condition
    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      baseIds.clear();

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      sideType = Teuchos::rcp(new shards::CellTopology(cellType->getCellTopologyData()->side[elem_side].topology));
      Intrepid::DefaultCubatureFactory<RealType> cubFactory;
      cubatureSide = cubFactory.create(*sideType, cubatureDegree);
      sideDims = sideType->getDimension();
      numQPsSide = cubatureSide->getNumPoints();

      // Allocate Temporary FieldContainers
      cubPointsSide.resize(numQPsSide, sideDims);
      refPointsSide.resize(numQPsSide, cellDims);
      cubWeightsSide.resize(numQPsSide);
      physPointsSide.resize(1, numQPsSide, cellDims);
      dofSide.resize(1, numQPsSide);
      dofSideVec.resize(1, numQPsSide, numVecDims);

      // Do the BC one side at a time for now
      jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
      invJacobianSide.resize(1, numQPsSide, cellDims, cellDims);
      jacobianSide_det.resize(1, numQPsSide);

      weighted_measure.resize(1, numQPsSide);
      basis_refPointsSide.resize(numNodes, numQPsSide);
      basisGrad_refPointsSide.resize(numNodes, numQPsSide, cellDims);
      trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
      trans_gradBasis_refPointsSide.resize(1, numNodes, numQPsSide, cellDims);
      weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);

      // Pre-Calculate reference element quantitites
      cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

      H_Side.resize(numQPsSide);
      H0_Side.resize(numQPsSide);
      V_Side.resize(numQPsSide, numVecDims);

      // Copy the coordinate data over to a temp container
     const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];

     for (std::size_t node = 0; node < numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        if(ilayer == numLayers)
          baseIds[base_id] = node;

        for (std::size_t dim = 0; dim < cellDims; ++dim)
          physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);

        if(ilayer==numLayers)
          physPointsCell(0, node, cellDims-1) = 0.0;
        else
          physPointsCell(0, node, cellDims-1) = -1.0;
      }

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid::CellTools<RealType>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate side geometry
      Intrepid::CellTools<MeshScalarT>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid::CellTools<MeshScalarT>::setJacobianInv(invJacobianSide, jacobianSide);

      Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);

      if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
        Intrepid::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      } else { //for 3D, get weighted face measure
        Intrepid::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid::OPERATOR_VALUE);

      intrepidBasis->getValues(basisGrad_refPointsSide, refPointsSide, Intrepid::OPERATOR_GRAD);

      // Transform values of the basis functions
      Intrepid::FunctionSpaceTools::HGRADtransformVALUE<MeshScalarT>(trans_basis_refPointsSide, basis_refPointsSide);

      Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>(trans_gradBasis_refPointsSide, invJacobianSide, basisGrad_refPointsSide);

      // Multiply with weighted measure
      Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
      Intrepid::CellTools<MeshScalarT>::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, *cellType);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      Intrepid::FieldContainer<ScalarT> H_Cell(numNodes);
      Intrepid::FieldContainer<ScalarT> H0_Cell(numNodes);
      Intrepid::FieldContainer<ScalarT> V_Cell(numNodes, numVecDims);
      Intrepid::FieldContainer<ScalarT> gradH_Side(numQPsSide, numVecDims);
      Intrepid::FieldContainer<ScalarT> divV_Side(numQPsSide);

      for (int i = 0; i < gradH_Side.size(); i++)
        gradH_Side(i) = 0.0;

      for (int i = 0; i < divV_Side.size(); i++)
        divV_Side(i) = 0.0;

      std::map<LO, std::size_t>::const_iterator it;


      for (it = baseIds.begin(); it != baseIds.end(); ++it){
        std::size_t node = it->second;
        H_Cell(node) = H(elem_LID, node);
        H0_Cell(node) = H0(elem_LID, node);
        for (std::size_t dim = 0; dim < numVecDims; ++dim)
          V_Cell(node, dim) = V(elem_LID, node, dim);
      }

      // This is needed, since evaluate currently sums into
      for (int qp = 0; qp < numQPsSide; qp++) {
        H_Side(qp) = 0.0;
        H0_Side(qp) = 0.0;
        for (std::size_t dim = 0; dim < numVecDims; ++dim)
          V_Side(qp, dim) = 0.0;
      }

//      std::cout << "At: "<<__LINE__ <<std::endl;
      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (it = baseIds.begin(); it != baseIds.end(); ++it){
        std::size_t node = it->second;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          const MeshScalarT& tmp = trans_basis_refPointsSide(0, node, qp);
       //   if(fabs(tmp)<1e-5) continue;
          H_Side(qp) += H_Cell(node) * tmp;
          H0_Side(qp) += H0_Cell(node) * tmp;
          for (std::size_t dim = 0; dim < numVecDims; ++dim)
            V_Side(qp, dim) += V_Cell(node, dim) * tmp;
        }
      }
  //    std::cout << "At: "<<__LINE__ <<std::endl;

      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (it = baseIds.begin(); it != baseIds.end(); ++it){
          std::size_t node = it->second;
          for (std::size_t dim = 0; dim < numVecDims; ++dim) {
            const MeshScalarT& tmp = trans_gradBasis_refPointsSide(0, node, qp, dim);
        //    if(fabs(tmp)<1e-7) continue;
            gradH_Side(qp, dim) += H0_Cell(node) * tmp;
            divV_Side(qp) += V_Cell(node, dim) * tmp;
          }
        }
      }
    //  std::cout << "At: "<<__LINE__ <<std::endl;


      for (it = baseIds.begin(); it != baseIds.end(); ++it){
        std::size_t node = it->second;
        ScalarT res = 0;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
         // std::cout << "H- " << H_Side(qp) << std::endl;
          ScalarT divHV = divV_Side(qp)* H0_Side(qp);
          for (std::size_t dim = 0; dim < numVecDims; ++dim)
            divHV += gradH_Side(qp, dim)*V_Side(qp,dim);

          ScalarT tmp = H_Side(qp)-H0_Side(qp) + (dt/1000.0) * divHV;
          res +=tmp * weighted_trans_basis_refPointsSide(0, node, qp);
        }
        Residual(elem_LID,node) = res;
      }
    }
 //   std::cout << std::endl;
 //   std::cout << std::endl;
  }
}

//**********************************************************************
}

