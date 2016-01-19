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
FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    coordVec("Coord Vec", dl->vertices_vector), surfaceVelocity_field("surface_velocity", dl->node_vector), basal_friction_field("basal_friction", dl->node_scalar), velocityRMS_field("surface_velocity_rms", dl->node_vector),
    velocity_field(p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get("Velocity Name", "Velocity"), dl->node_vector), numVecDim(2) {
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

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
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
  dofCellVec.resize(1, numNodes, numVecDim);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs = dims[1];
  numDims = dims[2];

  // add dependent fields
  this->addDependentField(velocity_field);
  this->addDependentField(surfaceVelocity_field);
  this->addDependentField(velocityRMS_field);
  this->addDependentField(coordVec);
  this->addDependentField(basal_friction_field);

  this->setName(fieldName + " Response surface_velocity Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response surface_velocity Mismatch";
  std::string global_response_name = fieldName + " Global Response surface_velocity Mismatch";
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
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(velocity_field, fm);
  this->utils.setFieldData(surfaceVelocity_field, fm);
  this->utils.setFieldData(velocityRMS_field, fm);
  this->utils.setFieldData(basal_friction_field, fm);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset) {
  PHAL::set(this->global_response, 0.0);

  p_resp = p_reg = 0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset) {
  if (workset.sideSets == Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find("upperside");

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> surfaceVelocityOnSide;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocityRMSOnSide;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocityOnSide;

    // Zero out local response
    PHAL::set(this->local_response, 0.0);

    // Loop over the sides that form the boundary condition
    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      sideType = Teuchos::rcp(new shards::CellTopology(cellType->getCellTopologyData()->side[elem_side].topology));
      Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
      cubatureSide = cubFactory.create(*sideType, cubatureDegree);
      sideDims = sideType->getDimension();
      numQPsSide = cubatureSide->getNumPoints();

      // Allocate Temporary FieldContainers
      cubPointsSide.resize(numQPsSide, sideDims);
      refPointsSide.resize(numQPsSide, cellDims);
      cubWeightsSide.resize(numQPsSide);
      physPointsSide.resize(1, numQPsSide, cellDims);
      dofSide.resize(1, numQPsSide);
      dofSideVec.resize(1, numQPsSide, numVecDim);

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
      data.resize(1, numQPsSide);

      // Pre-Calculate reference element quantitites
      cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

      surfaceVelocityOnSide.resize(1, numQPsSide, numVecDim);
      velocityRMSOnSide.resize(1, numQPsSide, numVecDim);
      velocityOnSide.resize(1, numQPsSide, numVecDim);

      // Copy the coordinate data over to a temp container
      for (std::size_t node = 0; node < numNodes; ++node) {
        for (std::size_t dim = 0; dim < cellDims; ++dim)
          physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
      }

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<RealType>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate side geometry
      Intrepid2::CellTools<MeshScalarT>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid2::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);

      if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
        Intrepid2::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      } else { //for 3D, get weighted face measure
        Intrepid2::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      // Transform values of the basis functions
      Intrepid2::FunctionSpaceTools::HGRADtransformVALUE<MeshScalarT>(trans_basis_refPointsSide, basis_refPointsSide);

      // Multiply with weighted measure
      Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
      Intrepid2::CellTools<RealType>::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, intrepidBasis);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> surfaceVelocityOnCell(1, numNodes, numVecDim);
      Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocityRMSOnCell(1, numNodes, numVecDim);
      Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocityOnCell(1, numNodes, numVecDim);
      for (std::size_t node = 0; node < numNodes; ++node)
        for (std::size_t dim = 0; dim < numVecDim; ++dim) {
          surfaceVelocityOnCell(0, node, dim) = surfaceVelocity_field(elem_LID, node, dim);
          velocityRMSOnCell(0, node, dim) = velocityRMS_field(elem_LID, node, dim);
          dofCellVec(0, node, dim) = velocity_field(elem_LID, node, dim);
        }
        // This is needed, since evaluate currently sums into
        for (int i = 0; i < numQPsSide; i++) {
          for (std::size_t dim = 0; dim < numVecDim; ++dim) {
            surfaceVelocityOnSide(0, i, dim) = 0.0;
            velocityRMSOnSide(0, i, dim) = 0.0;
          }
        }
        for (int i = 0; i < dofSideVec.size(); i++)
          dofSideVec[i] = 0.0;

        // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
        for (std::size_t node = 0; node < numNodes; ++node) {
          for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
            for (std::size_t dim = 0; dim < numVecDim; ++dim) {
              surfaceVelocityOnSide(0, qp, dim) += surfaceVelocityOnCell(0, node, dim) * trans_basis_refPointsSide(0, node, qp);
              velocityRMSOnSide(0, qp, dim) += velocityRMSOnCell(0, node, dim) * trans_basis_refPointsSide(0, node, qp);
              dofSideVec(0, qp, dim) += dofCellVec(0, node, dim) * trans_basis_refPointsSide(0, node, qp);
            }
          }
        }

      int numCells = data.dimension(0); // How many cell's worth of data is being computed?
      int numPoints = data.dimension(1); // How many QPs per cell?

      for (int cell = 0; cell < numCells; cell++) {
        for (int pt = 0; pt < numPoints; pt++) {
          ScalarT refVel0 = asinh(surfaceVelocityOnSide(cell, pt, 0) / velocityRMSOnSide(cell, pt, 0) / asinh_scaling);
          ScalarT refVel1 = asinh(surfaceVelocityOnSide(cell, pt, 1) / velocityRMSOnSide(cell, pt, 1) / asinh_scaling);
          ScalarT vel0 = asinh(dofSideVec(cell, pt, 0) / velocityRMSOnSide(cell, pt, 0) / asinh_scaling);
          ScalarT vel1 = asinh(dofSideVec(cell, pt, 1) / velocityRMSOnSide(cell, pt, 1) / asinh_scaling);
          data(cell, pt) = asinh_scaling * asinh_scaling * ((refVel0 - vel0) * (refVel0 - vel0) + (refVel1 - vel1) * (refVel1 - vel1));
        }
      }

      ScalarT t = 0;
      for (std::size_t node = 0; node < numNodes; ++node)
        for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
          t += data(0, qp) * weighted_trans_basis_refPointsSide(0, node, qp);
        }

      this->local_response(elem_LID, 0) += t*scaling;
      this->global_response(0) += t*scaling;
      p_resp += t*scaling;
    }
  }


  //Regularization term on the basal side
  Albany::SideSetList::const_iterator ib = ssList.find("basalside");

  if (ib != ssList.end() && (alpha != 0)) {
    const std::vector<Albany::SideStruct>& sideSet = ib->second;

    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> basalFrictionOnSide(1, numQPsSide);
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> basalFrictionGradOnSide(1, numQPsSide, cellDims);
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> basalFrictionGradOnSideT(1, numQPsSide, cellDims);

    // Loop over the sides that form the boundary condition
    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      // Copy the coordinate data over to a temp container
      for (std::size_t node = 0; node < numNodes; ++node) {
        for (std::size_t dim = 0; dim < cellDims; ++dim)
          physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
      }

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

      Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> uTan(1, numQPsSide, cellDims), vTan(1, numQPsSide, cellDims);
      Intrepid2::CellTools<MeshScalarT>::getPhysicalFaceTangents(uTan, vTan,jacobianSide,elem_side,*cellType);

      // Multiply with weighted measure
      Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

      // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
      Intrepid2::CellTools<RealType>::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, intrepidBasis);

      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      Intrepid2::FieldContainer_Kokkos<ParamScalarT, PHX::Layout, PHX::Device> basalFrictionOnCell(1, numNodes);

      for (std::size_t node = 0; node < numNodes; ++node) {
        basalFrictionOnCell(0,node) = basal_friction_field(elem_LID, node);
      //  std::cout << basalFrictionOnCell(0,node)<< " ";
      }
      // This is needed, since evaluate currently sums into
      for (int qp = 0; qp < numQPsSide; qp++) {
        basalFrictionOnSide(0,qp) = 0.0;
        for (std::size_t dim = 0; dim < cellDims; ++dim) {
          basalFrictionGradOnSide(0,qp,dim) = 0;
        basalFrictionGradOnSideT(0,qp,dim) = 0;
        }
      }

      for (int i = 0; i < dofSideVec.size(); i++)
        dofSideVec[i] = 0.0;

      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      for (std::size_t qp = 0; qp < numQPsSide; ++qp) {
        for (std::size_t node = 0; node < numNodes; ++node)
          for (std::size_t dim = 0; dim < cellDims; ++dim)
            basalFrictionGradOnSide(0,qp,dim) += basalFrictionOnCell(0,node) * trans_gradBasis_refPointsSide(0, node, qp, dim);

        for (std::size_t dim = 0; dim < cellDims; ++dim) {
          basalFrictionGradOnSideT(0,qp,0) += basalFrictionGradOnSide(0,qp,dim)*uTan(0,qp,dim);
          basalFrictionGradOnSideT(0,qp,1) += basalFrictionGradOnSide(0,qp,dim)*vTan(0,qp,dim);
        }
      }

      int numCells = data.dimension(0); // How many cell's worth of data is being computed?
      int numPoints = data.dimension(1); // How many QPs per cell?

      for (int cell = 0; cell < numCells; cell++) {
        for (int pt = 0; pt < numPoints; pt++) {
          ScalarT sum=0;
         for (std::size_t dim = 0; dim < 2; ++dim)
            sum += std::pow(basalFrictionGradOnSideT(0,pt,dim),2.0);
          data(cell, pt) = sum;
        }
      }

      ScalarT t = 0;
      for (std::size_t qp = 0; qp < numQPsSide; ++qp)
        t += data(0, qp) * weighted_measure(0, qp);

      this->local_response(elem_LID, 0) += t*scaling*alpha;//*50.0;
      this->global_response(0) += t*scaling*alpha;//*50.0;
      p_reg += t*scaling*alpha;
    }
  }
  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {
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
Teuchos::RCP<const Teuchos::ParameterList> FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::getValidResponseParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseSurfaceVelocityMismatch Params"));
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

