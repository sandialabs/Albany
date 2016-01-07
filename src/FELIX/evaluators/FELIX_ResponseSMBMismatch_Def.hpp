//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
FELIX::ResponseSMBMismatch<EvalT, Traits>::ResponseSMBMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  H("Thickness", dl->node_scalar),
  velocity_field("Averaged Velocity", dl->node_vector),
  //SMB("SMB", dl->node_scalar),
  coordVec("Coord Vec", dl->vertices_vector),  numVecFODims(2) {

  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  std::string fieldName ="";
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  asinh_scaling = plist->get<double>("Asinh Scaling", 10.0);

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = paramList->get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  int position;

  // Build element and side integration support

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepid2Basis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology(elem_top));

  Intrepid2::DefaultCubatureFactory<RealType> cubFactory;
  cubatureCell = cubFactory.create(*cellType, 1); //meshSpecs->cubatureDegree);
  cubatureDegree = plist->isParameter("Cubature Degree") ? plist->get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  int containerSize = dim[0];
  numQPs = dim[1];
  cellDims = dim[2];

  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  dofCellVec.resize(1, numNodes, numVecFODims);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs = dims[1];
  numDims = dims[2];

  // add dependent fields
  this->addDependentField(H);
  this->addDependentField(velocity_field);
  //this->addDependentField(SMB);
  this->addDependentField(coordVec);


  this->setName(fieldName + " Response surface_mass_balace Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response surface_mass_balance Mismatch";
  std::string global_response_name = fieldName + " Global Response surface_mass_balance Mismatch";
  int worksetSize = dl->qp_scalar->dimension(0);
  int responseSize = 1;
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::setup(p, dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(H, fm);
  this->utils.setFieldData(velocity_field, fm);
  //this->utils.setFieldData(SMB, fm);
  this->utils.setFieldData(coordVec, fm);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset) {
  PHAL::set(this->global_response, 0.0);

  p_resp = p_reg = 0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset) {
  if (workset.sideSets == Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find("upperside");

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    Intrepid2::FieldContainer<ScalarT> H_Side;
    Intrepid2::FieldContainer<ScalarT> SMB_Side;
    Intrepid2::FieldContainer<ScalarT> V_Side;

    // Loop over the sides that form the boundary condition
    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[iSide].elem_GID;
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;

      const CellTopologyData_Subcell& side =  cellType->getCellTopologyData()->side[elem_side];
      sideType = Teuchos::rcp(new shards::CellTopology(side.topology));
      int numSideNodes = sideType->getNodeCount();
      Intrepid2::DefaultCubatureFactory<RealType> cubFactory;
      cubatureSide = cubFactory.create(*sideType, cubatureDegree);
      sideDims = sideType->getDimension();
      numQPsSide = cubatureSide->getNumPoints();

      // Allocate Temporary FieldContainers
      cubPointsSide.resize(numQPsSide, sideDims);
      refPointsSide.resize(numQPsSide, cellDims);
      cubWeightsSide.resize(numQPsSide);
      physPointsSide.resize(1, numQPsSide, cellDims);
      dofSide.resize(1, numQPsSide);
      dofSideVec.resize(1, numQPsSide, numVecFODims);

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
      SMB_Side.resize(numQPsSide);
      V_Side.resize(numQPsSide, numVecFODims);

      // Copy the coordinate data over to a temp container
     for (std::size_t node = 0; node < numNodes; ++node) {
       for (std::size_t dim = 0; dim < cellDims; ++dim)
         physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
       physPointsCell(0, node, cellDims-1) = -1.0; //set z=-1 on internal cell nodes and z=0 side (see next lines).
     }
     for (int i = 0; i < numSideNodes; ++i)
       physPointsCell(0, side.node[i], cellDims-1) = 0.0;  //set z=0 on side

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<RealType>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate side geometry
      Intrepid2::CellTools<MeshScalarT>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid2::CellTools<MeshScalarT>::setJacobianInv(invJacobianSide, jacobianSide);

      Intrepid2::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);

      if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
        Intrepid2::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      } else { //for 3D, get weighted face measure
        Intrepid2::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      intrepidBasis->getValues(basisGrad_refPointsSide, refPointsSide, Intrepid2::OPERATOR_GRAD);

      // Transform values of the basis functions
      Intrepid2::FunctionSpaceTools::HGRADtransformVALUE<MeshScalarT>(trans_basis_refPointsSide, basis_refPointsSide);

      Intrepid2::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>(trans_gradBasis_refPointsSide, invJacobianSide, basisGrad_refPointsSide);

      // Multiply with weighted measure
      Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
      Intrepid2::CellTools<MeshScalarT>::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, *cellType);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      Intrepid2::FieldContainer<ScalarT> H_Cell(numNodes);
      Intrepid2::FieldContainer<ScalarT> SMB_Cell(numNodes);
      Intrepid2::FieldContainer<ScalarT> V_Cell(numNodes, numVecFODims);
      Intrepid2::FieldContainer<ScalarT> gradH_Side(numQPsSide, numVecFODims);
      Intrepid2::FieldContainer<ScalarT> divV_Side(numQPsSide);

      for (int i = 0; i < gradH_Side.size(); i++)
        gradH_Side(i) = 0.0;

      for (int i = 0; i < divV_Side.size(); i++)
        divV_Side(i) = 0.0;

      std::map<LO, std::size_t>::const_iterator it;


      for (int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i]; //it->second;
        H_Cell(node) = H(elem_LID, node);
        SMB_Cell(node) = 0;//SMB(elem_LID, node);
        for (std::size_t dim = 0; dim < numVecFODims; ++dim)
          V_Cell(node, dim) = velocity_field(elem_LID, node, dim);
      }

      // This is needed, since evaluate currently sums into
      for (int qp = 0; qp < numQPsSide; qp++) {
        H_Side(qp) = 0.0;
        SMB_Side(qp) = 0.0;
        for (std::size_t dim = 0; dim < numVecFODims; ++dim)
          V_Side(qp, dim) = 0.0;
      }

      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i]; //it->second;
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          const MeshScalarT& tmp = trans_basis_refPointsSide(0, node, qp);
          H_Side(qp) += H_Cell(node) * tmp;
          SMB_Side(qp) += SMB_Cell(node) * tmp;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim)
            V_Side(qp, dim) += V_Cell(node, dim) * tmp;
        }
      }

      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (int i = 0; i < numSideNodes; ++i){
          std::size_t node = side.node[i]; //it->second;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            const MeshScalarT& tmp = trans_gradBasis_refPointsSide(0, node, qp, dim);
            gradH_Side(qp, dim) += H_Cell(node) * tmp;
            divV_Side(qp) += V_Cell(node, dim) * tmp;
          }
        }
      }

      ScalarT reg = 0;
      ScalarT res = 0;
      for (int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i]; //it->second;

        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          ScalarT divHV = divV_Side(qp)* H_Side(qp);
          ScalarT tmp=0;
          for (std::size_t dim = 0; dim < numVecFODims; ++dim) {
            divHV += gradH_Side(qp, dim)*V_Side(qp,dim);
            tmp += gradH_Side(qp, dim)*gradH_Side(qp, dim);
          }
          reg += alpha*tmp*weighted_trans_basis_refPointsSide(0, node, qp);

          ScalarT diff = divHV/1000.0 - SMB_Side(qp);
          res += diff*diff * weighted_trans_basis_refPointsSide(0, node, qp);
        }
      }
      res *= scaling;
      reg *= scaling;
      p_resp += res;
      p_reg += reg;
      this->local_response(elem_LID, 0) += res+reg;//*50.0;
      this->global_response(0) += res+reg;//*50.0;
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {
#if 0
  // Add contributions across processors
  Teuchos::RCP<Teuchos::ValueTypeSerializer<int, ScalarT> > serializer = workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM, partial_response.size(), &partial_response[0], &this->global_response[0]);
  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM,1, &p_resp, &resp);
  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM, 1, &p_reg, &reg);
#else
  //amb Deal with op[], pointers, and reduceAll.
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response);
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_resp);
  resp = p_resp;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg);
  reg = p_reg;
#endif

  if(workset.comm->getRank()   ==0)
    std::cout << "resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) << ", reg: " << Sacado::ScalarValue<ScalarT>::eval(reg) <<std::endl;

  if (rank(*workset.comm) == 0) {
    std::ofstream ofile;
    ofile.open("mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      //ofile << sqrt(this->global_response[0]);
      PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
      ofile << sqrt(*gr);
      ofile.close();
    }
  }

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList> FELIX::ResponseSMBMismatch<EvalT, Traits>::getValidResponseParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseSMBMismatch Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL = PHAL::SeparableScatterScalarResponse<EvalT, Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Field Name", "Solution", "Not used");
  validPL->set<double>("Regularization Coefficient", 1.0, "Regularization Coefficient");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<double>("Asinh Scaling", 1.0, "Scaling s in asinh(s*x)/s. Used to penalize high values of velocity");
  validPL->set<int>("Cubature Degree", 3, "degree of cubature used to compute the velocity mismatch");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}
// **********************************************************************

