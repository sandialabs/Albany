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
#include "Intrepid_FunctionSpaceTools.hpp"

template<typename EvalT, typename Traits>
FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    coordVec("Coord Vec", dl->vertices_vector), surfaceVelocity_field("Surface Velocity", dl->node_vector), velocityRMS_field("Velocity RMS", dl->node_vector), velocity_field("Velocity", dl->node_vector), numVecDim(2) {
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  std::string fieldName;
  fieldName = plist->get<std::string>("Field Name", "");

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = paramList->get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  int position;

  // PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  // this->addEvaluatedField(fieldTag);

  // Build element and side integration support

  const CellTopologyData * const elem_top = &meshSpecs->ctd;
 // const CellTopologyData * const elem_top = shards::getCellTopologyData< shards::Tetrahedron<4> >(); //&meshSpecs->ctd;
 // const CellTopologyData * const elem_top = shards::getCellTopologyData<shards::Hexahedron<8> >(); //&meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepidBasis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology(elem_top));

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  cubatureCell = cubFactory.create(*cellType, 1); //meshSpecs->cubatureDegree);

  const CellTopologyData * const side_top = elem_top->side[0].topology;

  sideType = Teuchos::rcp(new shards::CellTopology(side_top));

  int cubatureDegree = plist->isParameter("Cubature Degree") ? plist->get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  cubatureSide = cubFactory.create(*sideType, cubatureDegree);

  sideDims = sideType->getDimension();
  numQPsSide = cubatureSide->getNumPoints();

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  int containerSize = dim[0];
  numQPs = dim[1];
  cellDims = dim[2];

  // Allocate Temporary FieldContainers
  cubPointsSide.resize(numQPsSide, sideDims);
  refPointsSide.resize(numQPsSide, cellDims);
  cubWeightsSide.resize(numQPsSide);
  physPointsSide.resize(1, numQPsSide, cellDims);
  dofSide.resize(1, numQPsSide);
  dofSideVec.resize(1, numQPsSide, numVecDim);

  // Do the BC one side at a time for now
  jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
  jacobianSide_det.resize(1, numQPsSide);

  weighted_measure.resize(1, numQPsSide);
  basis_refPointsSide.resize(numNodes, numQPsSide);
  trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
  weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);

  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  dofCellVec.resize(1, numNodes, numVecDim);
  data.resize(1, numQPsSide);

  // Pre-Calculate reference element quantitites
  cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs = dims[1];
  numDims = dims[2];

  // add dependent fields
  this->addDependentField(velocity_field);
  this->addDependentField(surfaceVelocity_field);
  this->addDependentField(velocityRMS_field);
  this->addDependentField(coordVec);

  this->setName(fieldName + " Response Surface Velocity Mismatch" + PHX::TypeString<EvalT>::value);

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response Surface Velocity Mismatch";
  std::string global_response_name = fieldName + " Global Response Surface Velocity Mismatch";
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
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset) {
  for (typename PHX::MDField<ScalarT>::size_type i = 0; i < this->global_response.size(); i++)
    this->global_response[i] = 0.0;

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

  if (it == ssList.end())
    return; // This sideset does not exist in this workset (GAH - this can go away
            // once we move logic to BCUtils

  const std::vector<Albany::SideStruct>& sideSet = it->second;

  Intrepid::FieldContainer<ScalarT> surfaceVelocityOnSide(1, numQPsSide, numVecDim);
  Intrepid::FieldContainer<ScalarT> velocityRMSOnSide(1, numQPsSide, numVecDim);
  Intrepid::FieldContainer<ScalarT> velocityOnSide(1, numQPsSide, numVecDim);

  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i = 0; i < this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  // Loop over the sides that form the boundary condition
  for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

    // Get the data that corresponds to the side

    const int elem_GID = sideSet[side].elem_GID;
    const int elem_LID = sideSet[side].elem_LID;
    const int elem_side = sideSet[side].side_local_id;

    // Copy the coordinate data over to a temp container

    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t dim = 0; dim < cellDims - 1; ++dim)
        physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
      physPointsCell(0, node, 2) = 0;
    }

    // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
    Intrepid::CellTools<RealType>::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

    // Calculate side geometry
    Intrepid::CellTools<RealType>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

    Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);

    if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
      Intrepid::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
    } else { //for 3D, get weighted face measure
      Intrepid::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>(weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
    }

    // Values of the basis functions at side cubature points, in the reference parent cell domain
    intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid::OPERATOR_VALUE);

    // Transform values of the basis functions
    Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>(trans_basis_refPointsSide, basis_refPointsSide);

    // Multiply with weighted measure
    Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

    // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
    Intrepid::CellTools<RealType>::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, *cellType);

    // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
    Intrepid::FieldContainer<MeshScalarT> surfaceVelocityOnCell(1, numNodes, numVecDim);
    Intrepid::FieldContainer<MeshScalarT> velocityRMSOnCell(1, numNodes, numVecDim);
    Intrepid::FieldContainer<ScalarT> velocityOnCell(1, numNodes, numVecDim);
    for (std::size_t node = 0; node < numNodes; ++node) {
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

    }

    int numCells = data.dimension(0); // How many cell's worth of data is being computed?
    int numPoints = data.dimension(1); // How many QPs per cell?

    //std::cout << "DEBUG: applying const dudn to sideset " << this->sideSetID << ": " << (const_val * scale) << std::endl;

    //Intrepid::FieldContainer<MeshScalarT> side_normals(numCells, numPoints, cellDims);
    //Intrepid::FieldContainer<MeshScalarT> normal_lengths(numCells, numPoints);

    // for this side in the reference cell, get the components of the normal direction vector
    //Intrepid::CellTools<MeshScalarT>::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

    // scale normals (unity)
    //Intrepid::RealSpaceTools<MeshScalarT>::vectorNorm(normal_lengths, side_normals, Intrepid::NORM_TWO);
    //Intrepid::FunctionSpaceTools::scalarMultiplyDataData<MeshScalarT>(side_normals, normal_lengths, side_normals, true);

    double factor = 1.0;
    for (int cell = 0; cell < numCells; cell++) {
      for (int pt = 0; pt < numPoints; pt++) {
        ScalarT refVel0 = std::asinh(surfaceVelocityOnSide(cell, pt, 0) / velocityRMSOnSide(cell, pt, 0) / factor);
        ScalarT refVel1 = std::asinh(surfaceVelocityOnSide(cell, pt, 1) / velocityRMSOnSide(cell, pt, 1) / factor);
        ScalarT vel0 = std::asinh(dofSideVec(cell, pt, 0) / velocityRMSOnSide(cell, pt, 0) / factor);
        ScalarT vel1 = std::asinh(dofSideVec(cell, pt, 1) / velocityRMSOnSide(cell, pt, 1) / factor);
        data(cell, pt) = factor * factor * ((refVel0 - vel0) * (refVel0 - vel0) + (refVel1 - vel1) * (refVel1 - vel1));
      }
    }
    //  }

    ScalarT t = 0;
    for (std::size_t node = 0; node < numNodes; ++node)
      for (std::size_t qp = 0; qp < numQPsSide; ++qp)
        t += data(0, qp) * weighted_trans_basis_refPointsSide(0, node, qp);

    this->local_response(elem_LID, 0) += t;
    this->global_response(0) += t;

  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {
  // Add contributions across processors
  Teuchos::RCP<Teuchos::ValueTypeSerializer<int, ScalarT> > serializer = workset.serializerManager.template getValue<EvalT>();
  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM, this->global_response.size(), &this->global_response[0], &this->global_response[0]);

  if (rank(*workset.comm) == 0) {
    std::ofstream ofile;
    ofile.open("mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      ofile << sqrt(this->global_response[0]);
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
  validPL->set<int>("Cubature Degree", 3, "degree of cubature used to compute the velocity mismatch");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

