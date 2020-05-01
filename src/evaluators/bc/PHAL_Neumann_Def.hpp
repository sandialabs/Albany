//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "PHAL_Neumann.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace PHAL {

constexpr double pi = 3.1415926535897932385;

//**********************************************************************
template<typename EvalT, typename Traits>
NeumannBase<EvalT, Traits>::
NeumannBase(const Teuchos::ParameterList& p) :

  dl             (p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  meshSpecs      (p.get<Teuchos::RCP<Albany::MeshSpecsStruct> >("Mesh Specs Struct")),
  offset         (p.get<Teuchos::Array<int> >("Equation Offset")),
  sideSetID      (p.get<std::string>("Side Set ID")),
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector)
{
  // the input.xml string "NBC on SS sidelist_12 for DOF T set dudn" (or something like it)
  name = p.get< std::string >("Neumann Input String");

  // The input.xml argument for the above string
  inputValues = p.get<Teuchos::Array<double> >("Neumann Input Value");

  // The input.xml argument for the above string
  inputConditions = p.get< std::string >("Neumann Input Conditions");

  // The DOF offsets are contained in the Equation Offset array. The length of this array are the
  // number of DOFs we will set each call
  numDOFsSet = offset.size();

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library");

  // If we are doing a Neumann internal boundary with a "scaled jump",
  // build a scale lookup table from the materialDB file (this must exist)

  int position;

  // User has specified conditions on sideset normal
  if(inputConditions == "scaled jump") {
    bc_type = INTJUMP;
    const_val = inputValues[0];
    this->registerSacadoParameter(name, paramLib);

    // Build a vector to hold the scaling from the material DB
    matScaling.resize(meshSpecs->ebNameToIndex.size(),1.0);

    //! Material database - holds the scaling we need
    Teuchos::RCP<Albany::MaterialDatabase> materialDB = p.get< Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB");

    // iterator over all ebnames in the mesh
    std::map<std::string, int>::const_iterator it;
    for(it = meshSpecs->ebNameToIndex.begin(); it != meshSpecs->ebNameToIndex.end(); it++){
      TEUCHOS_TEST_FOR_EXCEPTION(!materialDB->isElementBlockParam(it->first, "Flux Scale"),
                                  Teuchos::Exceptions::InvalidParameter,
                                  "Cannot locate the value of \"Flux Scale\" for element block " << it->first << " in the material database");

      matScaling[it->second] = materialDB->getElementBlockParam<double>(it->first, "Flux Scale");
    }
  } else if(inputConditions == "robin" || inputConditions == "radiate") {

    bc_type = (inputConditions == "radiate" ? STEFAN_BOLTZMANN : ROBIN);

    robin_vals[0] = inputValues[0]; // dof_value
    robin_vals[1] = inputValues[1]; // coeff multiplying difference 'dof^4 - dof_value^4' (or 'dof-dof_value' for robin)

    for(int i = 0; i < 2; i++) {
      std::stringstream ss; ss << name << "[" << i << "]";
      this->registerSacadoParameter(ss.str(), paramLib);
    }

    vectorDOF = p.get<bool>("Vector Field");

    dof = decltype(dof)(p.get<std::string>("DOF Name"),
                        p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
    this->addDependentField(dof);
  } else if( inputConditions == "closed_form") {
    bc_type = CLOSED_FORM;
  }

  // else parse the input to determine what type of BC to calculate

    // is there a "(" in the string?
  else if((position = inputConditions.find_first_of("(")) != std::string::npos){

      if(inputConditions.find("t_x", position + 1)){
        // User has specified conditions in base coords
        bc_type = TRACTION;
      }
      else {
        // User has specified conditions in base coords
        bc_type = COORD;
      }

      dudx.resize(meshSpecs->numDim);
      for(int i = 0; i < dudx.size(); i++)
        dudx[i] = inputValues[i];

      for(int i = 0; i < dudx.size(); i++) {
        std::stringstream ss; ss << name << "[" << i << "]";
        this->registerSacadoParameter(ss.str(), paramLib);
      }
  } else if(inputConditions == "P"){ // Pressure boundary condition for Elasticity

      // User has specified a pressure condition
      bc_type = PRESS;
      const_val = inputValues[0];
      this->registerSacadoParameter(name, paramLib);

  } else {

      // User has specified conditions on sideset normal
      bc_type = NORMAL;
      const_val = inputValues[0];
      this->registerSacadoParameter(name, paramLib);

  }

  this->addDependentField(coordVec);

  PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  this->addEvaluatedField(fieldTag);

  // Build element and side integration support

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepid2Basis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology (elem_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cubatureCell = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs->cubatureDegree);

  int cubatureDegree = (p.get<int>("Cubature Degree") > 0 ) ? p.get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  numSidesOnElem = elem_top->side_count;
  sideType.resize(numSidesOnElem);
  cubatureSide.resize(numSidesOnElem);
  side_type.resize(numSidesOnElem);

  // Build containers that depend on side topology
  const char* sideTypeName;

  maxSideDim = maxNumQpSide =0;
  for(int i=0; i<numSidesOnElem; ++i) {
    sideType[i] = Teuchos::rcp(new shards::CellTopology(elem_top->side[i].topology));
    cubatureSide[i] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[i], cubatureDegree);
    maxSideDim = std::max( maxSideDim, (int)sideType[i]->getDimension());
    maxNumQpSide = std::max(maxNumQpSide, (int)cubatureSide[i]->getNumPoints());
    sideTypeName = sideType[i]->getName();
    if(strncasecmp(sideTypeName, "Line", 4) == 0)
      side_type[i] = LINE;
    else if(strncasecmp(sideTypeName, "Tri", 3) == 0)
      side_type[i] = TRI;
    else if(strncasecmp(sideTypeName, "Quad", 4) == 0)
      side_type[i] = QUAD;
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "PHAL_Neumann: side type : " << sideTypeName << " is not supported." << std::endl);
  }

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  numCells = dim[0];
  numQPs = dim[1];
  cellDims = dim[2];


  this->setName(name+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  if (inputConditions == "robin" || inputConditions == "radiate")
  {
    this->utils.setFieldData(dof,fm);
    dofSide_buffer = Kokkos::createDynRankView(dof.get_view(), "dofSide", numCells*maxNumQpSide*numDOFsSet);
  }
  // Note, we do not need to add dependent field to fm here for output - that is done
  // by Neumann Aggregator

  // Allocate Temporary Views
  physPointsCell_buffer = Kokkos::createDynRankView(coordVec.get_view(), "physPointsCell", numCells*numNodes*cellDims);
  temporary_buffer = Kokkos::createDynRankView(coordVec.get_view(),"temporary_buffer", numCells*maxNumQpSide*cellDims*cellDims);

  cubPointsSide_buffer = Kokkos::DynRankView<RealType, PHX::Device>("cubPointsSide", maxNumQpSide*maxSideDim);
  refPointsSide_buffer = Kokkos::DynRankView<RealType, PHX::Device>("refPointsSide", maxNumQpSide*cellDims);
  cubWeightsSide_buffer = Kokkos::DynRankView<RealType, PHX::Device>("cubWeightsSide", maxNumQpSide);
  basis_refPointsSide_buffer = Kokkos::DynRankView<RealType, PHX::Device>("basis_refPointsSide", numNodes*maxNumQpSide);

  physPointsSide_buffer = Kokkos::createDynRankView(coordVec.get_view(),"physPointsSide", numCells*maxNumQpSide*cellDims);
  jacobianSide_buffer = Kokkos::createDynRankView(coordVec.get_view(),"jacobianSide", numCells*maxNumQpSide*cellDims*cellDims);
  jacobianSide_det_buffer = Kokkos::createDynRankView(coordVec.get_view(),"jacobianSide", numCells*maxNumQpSide);
  weighted_measure_buffer = Kokkos::createDynRankView(coordVec.get_view(),"weighted_measure", numCells*maxNumQpSide);
  trans_basis_refPointsSide_buffer = Kokkos::createDynRankView(coordVec.get_view(),"trans_basis_refPointsSide", numCells*numNodes*maxNumQpSide);
  weighted_trans_basis_refPointsSide_buffer = Kokkos::createDynRankView(coordVec.get_view(),"weighted_trans_basis_refPointsSide", numCells*numNodes*maxNumQpSide);
  side_normals_buffer = Kokkos::createDynRankView(coordVec.get_view(),"side_normals", numCells*maxNumQpSide*cellDims);
  normal_lengths_buffer =Kokkos::createDynRankView(coordVec.get_view(),"normal_lengths", numCells*maxNumQpSide);

  if (inputConditions == "robin" || inputConditions == "radiate") {
    dofCell_buffer = Kokkos::createDynRankView(dof.get_view(), "dofCell", numCells, numNodes, numDOFsSet);
  }

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
evaluateNeumannContribution(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  if(workset.sideSets == Teuchos::null || this->sideSetID.length() == 0)

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
         "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // neumann data type is always ScalarT, but the deriv dimension
  // actually needed depends on BC type. For many it just needs
  // deriv dimensions from MeshScalarT (cloned from coordVec).
  // "data" is same as neumann -- always ScalarT but not always
  // with full deriv dimension of a ScalarT variable.

  // std::cout << "NN0 " << std::endl;
  switch(bc_type){
    case INTJUMP:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case ROBIN:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case STEFAN_BOLTZMANN:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case NORMAL:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case PRESS:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case TRACTION:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    case CLOSED_FORM:
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (dof.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
    default:
    //std::cout << "NN1 " << std::endl;
       neumann = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
         (coordVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
       break;
  }

  data_buffer = Kokkos::createDynRankView(neumann, "data", numCells*maxNumQpSide*numDOFsSet);

  // Needed?
  Kokkos::deep_copy(neumann, 0.0);

  //std::cout << "NNN " << neumann(0,0,0) << std::endl;

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->sideSetID);

  const std::vector<Albany::SideStruct>& sideSet = it->second;

  if(it == ssList.end()) return; // This sideset does not exist in this workset (GAH - this can go away
                                  // once we move logic to BCUtils

  using DynRankViewRealT = Kokkos::DynRankView<RealType, PHX::Device>;
  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  using DynRankViewScalarT = Kokkos::DynRankView<ScalarT, PHX::Device>;

  DynRankViewRealT cubPointsSide;
  DynRankViewRealT refPointsSide;
  DynRankViewRealT cubWeightsSide;
  DynRankViewRealT basis_refPointsSide;

  DynRankViewMeshScalarT physPointsSide;
  DynRankViewMeshScalarT jacobianSide;
  DynRankViewMeshScalarT jacobianSide_det;
  DynRankViewMeshScalarT weighted_measure;
  DynRankViewMeshScalarT trans_basis_refPointsSide;
  DynRankViewMeshScalarT weighted_trans_basis_refPointsSide;
  DynRankViewMeshScalarT physPointsCell;

  DynRankViewScalarT dofSide;
  DynRankViewScalarT dofSideVec;
  DynRankViewScalarT dofCell;
  DynRankViewScalarT dofCellVec;

  DynRankViewScalarT data;

  //! For each element block, and for each local side id (e.g. side_id=0,1,2,3,4 for a Prism) we want to identify all the physical cells associated to that side id and block.
  //! In this way we can group them and call Intrepid2 function for a group of cells, which is more effective.
  //! At this point we do not know the number of blocks in this workset (If we assumed to have elements of the same block in a workset we could skip some of this).
  //! Also we do not know before the evaluator how many cells are associated to a local side id.

  std::map<int, int> ordinalEbIndex;
  std::vector<int> ebIndexVec;
  std::vector<std::vector<int> > numCellsOnSidesOnBlocks;
  std::vector<std::vector<Kokkos::DynRankView<int, PHX::Device> > > cellsOnSidesOnBlocks;
  for (auto const& it_side : sideSet) {
    const int ebIndex = it_side.elem_ebIndex;
    const int elem_side = it_side.side_local_id;

    if(ordinalEbIndex.insert(std::pair<int,int>(ebIndex,ordinalEbIndex.size())).second) {
      numCellsOnSidesOnBlocks.push_back(std::vector<int>(numSidesOnElem, 0));
      ebIndexVec.push_back(ebIndex);
    }

    numCellsOnSidesOnBlocks[ordinalEbIndex[ebIndex]][elem_side]++;
  }
  cellsOnSidesOnBlocks.resize(ordinalEbIndex.size());
  for (int ib=0; ib<ordinalEbIndex.size(); ib++) {
    cellsOnSidesOnBlocks[ib].resize(numSidesOnElem);
    for (int is=0; is<numSidesOnElem; is++) {
      cellsOnSidesOnBlocks[ib][is] = Kokkos::DynRankView<int, PHX::Device>("cellOnSide_i", numCellsOnSidesOnBlocks[ib][is]);
      numCellsOnSidesOnBlocks[ib][is]=0;
    }
  }

  for (auto const& it_side : sideSet) {
    const int iBlock = ordinalEbIndex[it_side.elem_ebIndex];
    const int elem_LID = it_side.elem_LID;
    const int elem_side = it_side.side_local_id;

    cellsOnSidesOnBlocks[iBlock][elem_side](numCellsOnSidesOnBlocks[iBlock][elem_side]++) = elem_LID;
  }

  // Loop over the sides that form the boundary condition
  for (int iblock = 0; iblock < ordinalEbIndex.size(); ++iblock)
  for (int side = 0; side < numSidesOnElem; ++side)
  {
    int numCells_ =  numCellsOnSidesOnBlocks[iblock][side];
    if( numCells_ == 0) continue;

    // Get the data that corresponds to the side

    int sideDims = sideType[side]->getDimension();
    int numQPsSide = cubatureSide[side]->getNumPoints();

    Kokkos::DynRankView<int, PHX::Device> cellVec  = cellsOnSidesOnBlocks[iblock][side];

    //need to resize containers because they depend on side topology
    cubPointsSide = DynRankViewRealT(cubPointsSide_buffer.data(), numQPsSide, sideDims);
    refPointsSide = DynRankViewRealT(refPointsSide_buffer.data(), numQPsSide, cellDims);
    cubWeightsSide = DynRankViewRealT(cubWeightsSide_buffer.data(), numQPsSide);
    basis_refPointsSide = DynRankViewRealT(basis_refPointsSide_buffer.data(), numNodes, numQPsSide);

    physPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(physPointsSide_buffer, physPointsSide_buffer.data(), numCells_, numQPsSide, cellDims);
    jacobianSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(jacobianSide_buffer, jacobianSide_buffer.data(), numCells_, numQPsSide, cellDims, cellDims);
    jacobianSide_det = Kokkos::createViewWithType<DynRankViewMeshScalarT>(jacobianSide_det_buffer, jacobianSide_det_buffer.data(), numCells_, numQPsSide);
    weighted_measure = Kokkos::createViewWithType<DynRankViewMeshScalarT>(weighted_measure_buffer, weighted_measure_buffer.data(), numCells_, numQPsSide);
    trans_basis_refPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(trans_basis_refPointsSide_buffer, trans_basis_refPointsSide_buffer.data(), numCells_, numNodes, numQPsSide);
    weighted_trans_basis_refPointsSide = Kokkos::createViewWithType<DynRankViewMeshScalarT>(weighted_trans_basis_refPointsSide_buffer, weighted_trans_basis_refPointsSide_buffer.data(), numCells_, numNodes, numQPsSide);
    physPointsCell =Kokkos::createViewWithType<DynRankViewMeshScalarT>(physPointsCell_buffer, physPointsCell_buffer.data(), numCells_, numNodes, cellDims);


    cubatureSide[side]->getCubature(cubPointsSide, cubWeightsSide);

    // Copy the coordinate data over to a temp container
    for (std::size_t iCell=0; iCell < numCells_; ++iCell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t dim=0; dim < cellDims; ++dim) {
          physPointsCell(iCell, node, dim) = coordVec(cellVec(iCell),node,dim);
    }}}

    // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
    ICT::mapToReferenceSubcell(refPointsSide, cubPointsSide, sideDims, side, *cellType);

    // Calculate side geometry
    ICT::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

    ICT::setJacobianDet(jacobianSide_det, jacobianSide);

    if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
      IFST::computeEdgeMeasure(weighted_measure, jacobianSide, cubWeightsSide, side, *cellType, temporary_buffer);
    } else { //for 3D, get weighted face measure
      IFST::computeFaceMeasure(weighted_measure, jacobianSide, cubWeightsSide, side, *cellType, temporary_buffer);
    }

    // Values of the basis functions at side cubature points, in the reference parent cell domain
    intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

    // Transform values of the basis functions
    IFST::HGRADtransformVALUE(trans_basis_refPointsSide, basis_refPointsSide);

    // Multiply with weighted measure
    IFST::multiplyMeasure(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

    // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
    ICT::mapToPhysicalFrame(physPointsSide, refPointsSide, physPointsCell, intrepidBasis);


    // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
    if(bc_type == ROBIN || bc_type == STEFAN_BOLTZMANN ) {
      dofCell = Kokkos::createViewWithType<DynRankViewScalarT>(dofCell_buffer, dofCell_buffer.data(), numCells_, numNodes, numDOFsSet);
      dofSide = Kokkos::createViewWithType<DynRankViewScalarT>(dofSide_buffer, dofSide_buffer.data(), numCells_, numQPsSide, numDOFsSet);

      Kokkos::deep_copy(dofCell, 0.0);
      for (std::size_t iCell=0; iCell < numCells_; ++iCell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t icomp=0; icomp < numDOFsSet; ++icomp) {
            if (vectorDOF) {
              dofCell(iCell, node, icomp) = dof(cellVec(iCell), node, this->offset[icomp]);
            } else {
              dofCell(iCell, node, icomp) = dof(cellVec(iCell), node);
            }
      }}}

      // This is needed, since evaluate currently sums into
      Kokkos::deep_copy(dofSide, 0.0);

      for (std::size_t icomp=0; icomp < numDOFsSet; ++icomp) {
        IFST::evaluate(Kokkos::subview(dofSide,Kokkos::ALL(),Kokkos::ALL(),icomp),
                       Kokkos::subview(dofCell,Kokkos::ALL(),Kokkos::ALL(),icomp),
                       trans_basis_refPointsSide);
      }
    }

    // Transform the given BC data to the physical space QPs in each side (elem_side)
    data = Kokkos::createViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >(data_buffer,data_buffer.data(), numCells_, numQPsSide, numDOFsSet);

    // Note: if you add a BC here, you need to add it above as well
    // to allocate neumann correctly.
    switch(bc_type){

      case INTJUMP:
       {
         const ScalarT elem_scale = matScaling[ebIndexVec[iblock]];
         calc_dudn_const(data, elem_scale);
         break;
       }

      case ROBIN:
         calc_dudn_robin(data, dofSide);
         break;

      case STEFAN_BOLTZMANN:
         calc_dudn_radiate(data, dofSide);
         break;

      case NORMAL:
         calc_dudn_const(data);
         break;

      case PRESS:
         calc_press(data, jacobianSide, *cellType, side);
         break;

      case TRACTION:

         calc_traction_components(data);
         break;
      case CLOSED_FORM:

         calc_closed_form(data, physPointsSide, jacobianSide, *cellType, side, workset);
         break;
      default:

         calc_gradu_dotn_const(data, jacobianSide, *cellType, side);
         break;
    }


    // Put this side's contribution into the vector
    for (std::size_t iCell=0; iCell < numCells_; ++iCell)
    {
      int cell = cellVec(iCell);
      for (std::size_t node=0; node < numNodes; ++node)
        for (std::size_t qp=0; qp < numQPsSide; ++qp)
          for (std::size_t dim=0; dim < numDOFsSet; ++dim)
            neumann(cell, node, dim) +=
                  data(iCell, qp, dim) * weighted_trans_basis_refPointsSide(iCell, node, qp);
    }
  }
}

template<typename EvalT, typename Traits>
typename NeumannBase<EvalT, Traits>::ScalarT&
NeumannBase<EvalT, Traits>::
getValue(const std::string &n) {

  if(std::string::npos != n.find("robin")) {
    for(int i = 0; i < 2; i++) {
      std::stringstream ss; ss << name << "[" << i << "]";
      if (n == ss.str())  return robin_vals[i];
    }
  }
  else if(std::string::npos != n.find("radiate")) {
    for(int i = 0; i < 2; i++) {
      std::stringstream ss; ss << name << "[" << i << "]";
      if (n == ss.str())  return robin_vals[i];
    }
  }
  else {
    for(int i = 0; i < dudx.size(); i++) {
      std::stringstream ss; ss << name << "[" << i << "]";
      if (n == ss.str())  return dudx[i];
    }
  }

//  if (n == name) return const_val;
  return const_val;

}


template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_traction_components(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned) const {

  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?

  for(int cell = 0; cell < numCells_; cell++)
    for(int pt = 0; pt < numPoints; pt++)
      for(int dim = 0; dim < numDOFsSet; dim++)
        qp_data_returned(cell, pt, dim) = -dudx[dim];

}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_gradu_dotn_const(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                          const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          int local_side_id) const {

  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?

  Kokkos::DynRankView<ScalarT, PHX::Device> grad_T =  Kokkos::createDynRankView(qp_data_returned, "grad_T", numCells_, numPoints, cellDims);
  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals = Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths = Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

/*
  double kdTdx[3];
  kdTdx[0] = 1.0; // Neumann component in the x direction
  kdTdx[1] = 0.0; // Neumann component in the y direction
  kdTdx[2] = 0.0; // Neumann component in the z direction
*/

  for(int side = 0; side < numCells_; side++) {
    for(int pt = 0; pt < numPoints; pt++) {
      for(int dim = 0; dim < cellDims; dim++) {
        grad_T(side, pt, dim) = dudx[dim]; // k grad T in the x direction goes in the x spot, and so on
  }}}

  // for this side in the reference cell, get the components of the normal direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell,local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData<MeshScalarT>(side_normals, normal_lengths, side_normals, true);

  // take grad_T dotted with the unit normal
  IFST::dotMultiplyDataData(qp_data_returned, grad_T, side_normals);
  // for(int cell = 0; cell < numCells; cell++)
  //   for(int pt = 0; pt < numPoints; pt++)
  //     for(int dim = 0; dim < numDOFsSet; dim++)
  //       qp_data_returned(cell, pt, dim) = grad_T(cell, pt, dim) * side_normals(cell, pt, dim);
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_dudn_const(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                ScalarT scale) const {

  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?

  //std::cout << "DEBUG: applying const dudn to sideset " << this->sideSetID << ": " << (const_val * scale) << std::endl;

  for(int side = 0; side < numCells_; side++) {
    for(int pt = 0; pt < numPoints; pt++) {
      for(int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(side, pt, dim) = -const_val * scale; // User directly specified dTdn, just use it
  }}}
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_dudn_robin(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const {

  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?

  const ScalarT& dof_value = robin_vals[0];
  const ScalarT& coeff = robin_vals[1];

  for(int side = 0; side < numCells_; side++) {
    for(int pt = 0; pt < numPoints; pt++) {
      for(int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(side, pt, dim) = coeff*(dof_side(side,pt,dim) - dof_value);
  }}}
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_dudn_radiate(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                const Kokkos::DynRankView<ScalarT, PHX::Device>& dof_side) const {

  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?

  const ScalarT& dof_value = robin_vals[0];
  const ScalarT dof_value4 = dof_value*dof_value*dof_value*dof_value; 
  const ScalarT& coeff = robin_vals[1];

   for (int cell = 0; cell < numCells_; cell++) {
    for (int pt = 0; pt < numPoints; pt++) {
      for (int dim = 0; dim < numDOFsSet; dim++) {
        qp_data_returned(cell, pt, dim) = coeff*(dof_side(cell,pt,dim)*dof_side(cell,pt,dim)*dof_side(cell,pt,dim)*dof_side(cell,pt,dim) - dof_value4);
      }
    }
  }
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_press(Kokkos::DynRankView<ScalarT, PHX::Device> & qp_data_returned,
                          const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          int local_side_id) const {

  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent(1); // How many QPs per cell?

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals = Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths = Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);

  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths, side_normals, true);

  for(int cell = 0; cell < numCells_; cell++)
    for(int pt = 0; pt < numPoints; pt++)
      for(int dim = 0; dim < numDOFsSet; dim++)
        qp_data_returned(cell, pt, dim) = const_val * side_normals(cell, pt, dim);

}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_closed_form(       Kokkos::DynRankView<ScalarT, PHX::Device>    & qp_data_returned,
                  const Kokkos::DynRankView<MeshScalarT, PHX::Device>& physPointsSide,
                  const Kokkos::DynRankView<MeshScalarT, PHX::Device>& jacobian_side_refcell,
                  const shards::CellTopology & celltopo,
                        int                    local_side_id,
               typename Traits::EvalData       workset) const
{
  // How many QPs per cell?
  int numCells_ = qp_data_returned.extent(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.extent( 1);

  using DynRankViewMeshScalarT = Kokkos::DynRankView<MeshScalarT, PHX::Device>;
  DynRankViewMeshScalarT side_normals =
          Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(side_normals_buffer, side_normals_buffer.data(), numCells_, numPoints, cellDims);
  DynRankViewMeshScalarT normal_lengths =
          Kokkos::createDynRankViewWithType<DynRankViewMeshScalarT>(normal_lengths_buffer, normal_lengths_buffer.data(), numCells_, numPoints);

  // for this side in the reference cell, get the components of the normal direction vector
  ICT::getPhysicalSideNormals(side_normals, jacobian_side_refcell, local_side_id, celltopo);
  // scale normals (unity)
  IRST::vectorNorm(normal_lengths, side_normals, Intrepid2::NORM_TWO);
  IFST::scalarMultiplyDataData(side_normals, normal_lengths,side_normals, true);

  for(int cell = 0; cell < numCells_; cell++)
  {
    for(int pt = 0; pt < numPoints; pt++)
    {
      MeshScalarT x = physPointsSide( cell, pt, 0);
      MeshScalarT y = physPointsSide( cell, pt, 1);
      MeshScalarT z = physPointsSide( cell, pt, 2);
      double      t = workset.current_time;
      for(int dim = 0; dim < numDOFsSet; dim++)
      {
        // Your closed form equation here!
        ScalarT value = 0.0*t + 0.0*x + 0.0*y + 0.0*z;
        qp_data_returned(cell, pt, dim) =  value;
      }
    }
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
Neumann<PHAL::AlbanyTraits::Residual,Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::Residual,Traits>(p)
{
}

template<typename Traits>
void Neumann<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_Vector> f = workset.f;
  Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(f);

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  // Place it at the appropriate offset into F
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
        f_nonconstView[nodeID(cell,node,this->offset[dim])] += this->neumann(cell, node, dim);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::Jacobian, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::Jacobian,Traits>(p)
{
}

// **********************************************************************
// #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void Neumann<PHAL::AlbanyTraits::Jacobian,Traits>::
// operator()(const Neumann_Tag& , const int& cell) const
// {
//   LO colT[1];
//   LO rowT;
//   ST value[1];
//   int lcol;
//   const int neq = Index.extent(2);
//   const int nunk = neq*this->numNodes;

//   for (std::size_t node = 0; node < this->numNodes; ++node) {
//     for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

//       int dim2=this->offset[dim];
//       rowT = Index(cell,node,dim2);

//       if (this->fT != Teuchos::null) {
//          this->fT->sumIntoLocalValue(rowT, this->neumann(cell, node, dim).val());
//       }

//       // Check derivative array is nonzero
//       if (this->neumann(cell, node, dim).hasFastAccess()) {
//         // Loop over nodes in element
//         for (unsigned int node_col=0; node_col<this->numNodes; node_col++){
//           // Loop over equations per node
//           for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
//             lcol = neq * node_col + eq_col;

//             // Global column
//             colT[0] =  Index(cell, node_col, eq_col);
//             value[0] = this->neumann(cell, node, dim).fastAccessDx(lcol);
//             if (is_adjoint) {
//               // Sum Jacobian transposed
//               jacobian.sumIntoValues(colT[0], &rowT,1, &value[0], false, true);
//             }
//             else {
//               // Sum Jacobian
//               jacobian.sumIntoValues(rowT, colT, nunk,value, false, true);
//             }
//           } // column equations
//         } // column nodes
//       } // has fast access
//     }
//   }
// }
// #endif

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
//IKT, 5/31/16: I commented out the KOKKOS_UNDER_DEVELOPMENT
//code b/c it does not execute correctly on an OpenMP KokkosNode.
//This problem shows up for some LandIce cases.
//It is somewhat of a mystery why this is the case b/c the Jacobian
//matrices dumped to matrix market _are_ correct.  This problem needs
//to be looked into.
//
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;

  Teuchos::RCP<Thyra_Vector>   f   = workset.f;
  Teuchos::RCP<Thyra_LinearOp> jac = workset.Jac;

  Teuchos::ArrayRCP<ST> f_nonconstView;
  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);
  int lcol;
  Teuchos::Array<LO> row(1);
  Teuchos::Array<LO> col(1);
  Teuchos::Array<ST> value(1);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

      row[0] = nodeID(cell,node,this->offset[dim]);

      int neq = nodeID.extent(2);

      if (f != Teuchos::null) {
        f_nonconstView[row[0]] += this->neumann(cell, node, dim).val();
      }

      // Check derivative array is nonzero
      if (this->neumann(cell, node, dim).hasFastAccess()) {

        // Loop over nodes in element
        for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

          // Loop over equations per node
          for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
            lcol = neq * node_col + eq_col;

            // Global column
            col[0] =  nodeID(cell,node_col,eq_col);
            value[0] = this->neumann(cell, node, dim).fastAccessDx(lcol);
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              Albany::addToLocalRowValues(jac,col[0],row(),value());
            } else {
              // Sum Jacobian
              Albany::addToLocalRowValues(jac,row[0],col(),value());
            }
          } // column equations
        } // column nodes
      } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::Tangent, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::Tangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;

  Teuchos::RCP<Thyra_Vector>       f = workset.f;
  Teuchos::RCP<Thyra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Thyra_MultiVector> fp = workset.fp;

  Teuchos::ArrayRCP<ST> f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fp_nonconst2dView;

  if (!f.is_null()) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (!JV.is_null()) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
  }
  if (!fp.is_null()) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        int row = nodeID(cell,node,this->offset[dim]);

        if (f != Teuchos::null) {
          f_nonconstView[row] += this->neumann(cell, node, dim).val();
        }

        if (JV != Teuchos::null) {
          for (int col=0; col<workset.num_cols_x; col++) {
            JV_nonconst2dView[col][row] += this->neumann(cell, node, dim).dx(col);
          }
        }
        if (fp != Teuchos::null) {
          for (int col=0; col<workset.num_cols_p; col++) {
            fp_nonconst2dView[col][row] += this->neumann(cell, node, dim).dx(col+workset.param_offset);
          }
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_MultiVector> fpV = workset.fpV;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);

  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.Vp->domain()->dim();

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if (trans) {
    int neq = workset.numEqs;
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];

    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];
      const int num_deriv = local_Vp.size()/neq;
      for (int i=0; i<num_deriv; i++) {
        const LO row = wsElDofs((int)cell,i,0);
        if(row<0) { continue; }
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
              int eq = this->offset[dim];
              val += this->neumann(cell, node, dim).dx(i)*local_Vp[node*neq+eq][col];
            }
          }
          fpV_nonconst2dView[col][row] += val;
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
          const int row = nodeID(cell,node,this->offset[dim]);
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i) {
              val += this->neumann(cell, node, dim).dx(i)*local_Vp[i][col];
            }
            fpV_nonconst2dView[col][row] += val;
          }
        }
      }
    }
  }
}

// **********************************************************************
// Simple evaluator to aggregate all Neumann BCs into one "field"
// **********************************************************************

template<typename EvalT, typename Traits>
NeumannAggregator<EvalT, Traits>::
NeumannAggregator(const Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> dl =  p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  const std::vector<std::string>& nbcs = *p.get<Teuchos::RCP<std::vector<std::string> > >("NBC Names");

  for (unsigned int i=0; i<nbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(nbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("NBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Neumann Aggregator"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void NeumannAggregator<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

} // namespace PHAL
