/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <string>

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Sacado_ParameterRegistration.hpp"


namespace PHAL {

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

  if((inputConditions == "scaled jump" || inputConditions == "robin") &&
     p.isType<Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB")){

    //! Material database - holds the scaling we need
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB =
      p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

     // User has specified conditions on sideset normal
    if(inputConditions == "scaled jump") {
      bc_type = INTJUMP;
      const_val = inputValues[0];
      new Sacado::ParameterRegistration<EvalT, SPL_Traits> (name, this, paramLib);
    }
    else { // inputConditions == "robin"
      bc_type = ROBIN;
      robin_vals[0] = inputValues[0]; // dof_value
      robin_vals[1] = inputValues[1]; // coeff multiplying difference (dof - dof_value) -- could be permittivity/distance (distance in mesh units)
      robin_vals[2] = inputValues[2]; // jump in slope (like plain Neumann bc)

      for(int i = 0; i < 3; i++) {
        std::stringstream ss; ss << name << "[" << i << "]";
        new Sacado::ParameterRegistration<EvalT, SPL_Traits> (ss.str(), this, paramLib);
      }
    }

     // Build a vector to hold the scaling from the material DB
     matScaling.resize(meshSpecs->ebNameToIndex.size());

     // iterator over all ebnames in the mesh

     std::map<std::string, int>::const_iterator it;
     for(it = meshSpecs->ebNameToIndex.begin(); it != meshSpecs->ebNameToIndex.end(); it++){

//      std::cout << "Searching for a value for \"Flux Scale\" in material database for element block: "
//        << it->first << std::endl;

       TEUCHOS_TEST_FOR_EXCEPTION(!materialDB->isElementBlockParam(it->first, "Flux Scale"),
         Teuchos::Exceptions::InvalidParameter, "Cannot locate the value of \"Flux Scale\" for element block " 
				  << it->first << " in the material database");

       matScaling[it->second] = 
         materialDB->getElementBlockParam<double>(it->first, "Flux Scale");

     }

     // In the robin boundary condition case, the NBC depends on the solution (dof) field
     if (inputConditions == "robin") {
      // Currently, the Neumann evaluator doesn't handle the case when the degree of freedom is a vector.
      // It wouldn't be difficult to have the boundary condition use a component of the vector, but I'm
      // not sure this is the correct behavior.  In any case, the only time when this evaluator needs
      // a degree of freedom value is in the "robin" case.
      TEUCHOS_TEST_FOR_EXCEPTION(p.get<bool>("Vector Field") == true, 
    			     Teuchos::Exceptions::InvalidParameter, 
    			     std::endl << "Error: \"Robin\" Neumann boundary conditions " 
			     << "only supported when the DOF is not a vector" << std::endl);

       PHX::MDField<ScalarT,Cell,Node> tmp(p.get<string>("DOF Name"),
           p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
       dof = tmp;

       this->addDependentField(dof);
     }
  }

  // else parse the input to determine what type of BC to calculate

    // is there a "(" in the string?
  else if(inputConditions.find_first_of("(") != string::npos){

      // User has specified conditions in base coords
      bc_type = COORD;
      dudx.resize(meshSpecs->numDim);
      for(int i = 0; i < dudx.size(); i++)
        dudx[i] = inputValues[i];

      for(int i = 0; i < dudx.size(); i++) {
        std::stringstream ss; ss << name << "[" << i << "]";
        new Sacado::ParameterRegistration<EvalT, SPL_Traits> (ss.str(), this, paramLib);
      }
  }
  else if(inputConditions == "P"){ // Pressure boundary condition for Elasticity

      // User has specified a pressure condition
      bc_type = PRESS;
      const_val = inputValues[0];
      new Sacado::ParameterRegistration<EvalT, SPL_Traits> (name, this, paramLib);

  }
  else {

      // User has specified conditions on sideset normal
      bc_type = NORMAL;
      const_val = inputValues[0];
      new Sacado::ParameterRegistration<EvalT, SPL_Traits> (name, this, paramLib);

  }

  this->addDependentField(coordVec);

  PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  this->addEvaluatedField(fieldTag);

  // Build element and side integration support

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepidBasis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology (elem_top));

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  cubatureCell = cubFactory.create(*cellType, meshSpecs->cubatureDegree);

  const CellTopologyData * const side_top = elem_top->side[0].topology;

  if(strncmp(side_top->name, "LINE", 4) == 0)

    side_type = LINE;

  else if(strncmp(side_top->name, "Tri", 3) == 0)

    side_type = TRI;

  else

    side_type = OTHER;

  sideType = Teuchos::rcp(new shards::CellTopology(side_top)); 
  cubatureSide = cubFactory.create(*sideType, meshSpecs->cubatureDegree);

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

  // Do the BC one side at a time for now
  jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
  jacobianSide_det.resize(1, numQPsSide);

  weighted_measure.resize(1, numQPsSide);
  basis_refPointsSide.resize(numNodes, numQPsSide);
  trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
  weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);

  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  neumann.resize(containerSize, numNodes, numDOFsSet);
  data.resize(1, numQPsSide, numDOFsSet);

  // Pre-Calculate reference element quantitites
  cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

  this->setName(name+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  if (inputConditions == "robin") this->utils.setFieldData(dof,fm);
  // Note, we do not need to add dependent field to fm here for output - that is done
  // by Neumann Aggregator
}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
evaluateNeumannContribution(typename Traits::EvalData workset)
{

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from 
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->sideSetID);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
   for (std::size_t node=0; node < numNodes; ++node)
     for (std::size_t dim=0; dim < numDOFsSet; ++dim)
	     neumann(cell, node, dim) = 0.0;

  if(it == ssList.end()) return; // This sideset does not exist in this workset (GAH - this can go away
                                  // once we move logic to BCUtils

  const std::vector<Albany::SideStruct>& sideSet = it->second;

  // Loop over the sides that form the boundary condition 

  for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

    // Get the data that corresponds to the side

    const int elem_GID = sideSet[side].elem_GID;
    const int elem_LID = sideSet[side].elem_LID;
    const int elem_side = sideSet[side].side_local_id;

    // Copy the coordinate data over to a temp container

    for (std::size_t node=0; node < numNodes; ++node)
      for (std::size_t dim=0; dim < cellDims; ++dim)
	physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);


    // Map side cubature points to the reference parent cell based on the appropriate side (elem_side) 
    Intrepid::CellTools<RealType>::mapToReferenceSubcell
      (refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

    // Calculate side geometry
    Intrepid::CellTools<RealType>::setJacobian
       (jacobianSide, refPointsSide, physPointsCell, *cellType);

    Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);

    // Get weighted edge measure
    Intrepid::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>
      (weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);

    // Values of the basis functions at side cubature points, in the reference parent cell domain
    intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid::OPERATOR_VALUE);

    // Transform values of the basis functions
    Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
      (trans_basis_refPointsSide, basis_refPointsSide);

    // Multiply with weighted measure
    Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
      (weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);

    // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
    Intrepid::CellTools<RealType>::mapToPhysicalFrame
      (physPointsSide, refPointsSide, physPointsCell, *cellType);

    
    // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
    if(bc_type == ROBIN) {
      for (std::size_t node=0; node < numNodes; ++node)
	dofCell(0, node) = dof(elem_LID, node);

      // This is needed, since evaluate currently sums into
      for (int i=0; i < numQPsSide ; i++) dofSide(0,i) = 0.0;

      // Get dof at cubature points of appropriate side (see DOFInterpolation evaluator)
      Intrepid::FunctionSpaceTools::
	evaluate<ScalarT>(dofSide, dofCell, trans_basis_refPointsSide);
    }

  // Transform the given BC data to the physical space QPs in each side (elem_side)

    switch(bc_type){
  
      case INTJUMP:
       {
         const ScalarT elem_scale = matScaling[sideSet[side].elem_ebIndex];
         calc_dudn_const(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side, elem_scale);
         break;
       }

      case ROBIN:
       {
         const ScalarT elem_scale = matScaling[sideSet[side].elem_ebIndex];
         calc_dudn_robin(data, physPointsSide, dofSide, jacobianSide, *cellType, cellDims, elem_side, elem_scale, robin_vals);
         break;
       }   
   
      case NORMAL:
  
         calc_dudn_const(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side);
         break;

      case PRESS:
  
         calc_press(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side);
         break;
  
      default:
  
         calc_gradu_dotn_const(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side);
         break;
  
    }

    // Put this side's contribution into the vector

    for (std::size_t node=0; node < numNodes; ++node)
      for (std::size_t qp=0; qp < numQPsSide; ++qp)
         for (std::size_t dim=0; dim < numDOFsSet; ++dim)
           neumann(elem_LID, node, dim) += 
                  data(0, qp, dim) * weighted_trans_basis_refPointsSide(0, node, qp);
  }
  
}

template<typename EvalT, typename Traits>
typename NeumannBase<EvalT, Traits>::ScalarT&
NeumannBase<EvalT, Traits>::
getValue(const std::string &n) {

  if (n == name) return const_val;
  else if(std::string::npos != n.find("robin")) {
    for(int i = 0; i < 3; i++) {
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
}


template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_gradu_dotn_const(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2); // How many DOFs per node to calculate?

  Intrepid::FieldContainer<ScalarT> grad_T(numCells, numPoints, cellDims);
  Intrepid::FieldContainer<MeshScalarT> side_normals(numCells, numPoints, cellDims);
  Intrepid::FieldContainer<MeshScalarT> normal_lengths(numCells, numPoints);

/*
  double kdTdx[3];
  kdTdx[0] = 1.0; // Neumann component in the x direction
  kdTdx[1] = 0.0; // Neumann component in the y direction
  kdTdx[2] = 0.0; // Neumann component in the z direction
*/

  for(int cell = 0; cell < numCells; cell++)
    for(int pt = 0; pt < numPoints; pt++)
      for(int dim = 0; dim < cellDims; dim++)
        grad_T(cell, pt, dim) = dudx[dim]; // k grad T in the x direction goes in the x spot, and so on

  // for this side in the reference cell, get the components of the normal direction vector
  Intrepid::CellTools<MeshScalarT>::getPhysicalSideNormals(side_normals, jacobian_side_refcell, 
    local_side_id, celltopo);

  // scale normals (unity)
  Intrepid::RealSpaceTools<MeshScalarT>::vectorNorm(normal_lengths, side_normals, Intrepid::NORM_TWO);
  Intrepid::FunctionSpaceTools::scalarMultiplyDataData<MeshScalarT>(side_normals, normal_lengths, 
    side_normals, true);

  // take grad_T dotted with the unit normal
//  Intrepid::FunctionSpaceTools::dotMultiplyDataData<ScalarT>(qp_data_returned, 
//    grad_T, side_normals);

  for(int pt = 0; pt < numPoints; pt++)
    for(int dim = 0; dim < numDOFsSet; dim++)
      qp_data_returned(0, pt, dim) = grad_T(0, pt, dim) * side_normals(0, pt, dim);

}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_dudn_const(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id,
                          ScalarT scale){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2); // How many DOFs per node to calculate?

  //std::cout << "DEBUG: applying const dudn to sideset " << this->sideSetID << ": " << (const_val * scale) << std::endl;

  for(int pt = 0; pt < numPoints; pt++)
    for(int dim = 0; dim < numDOFsSet; dim++)
      qp_data_returned(0, pt, dim) = -const_val * scale; // User directly specified dTdn, just use it


}

template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_dudn_robin(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
		const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
		const Intrepid::FieldContainer<ScalarT>& dof_side,
		const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
		const shards::CellTopology & celltopo,
		const int cellDims,
		int local_side_id,
		ScalarT scale,
		const ScalarT* robin_param_values){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2); // How many DOFs per node to calculate?

  const ScalarT& dof_value = robin_vals[0];
  const ScalarT& coeff = robin_vals[1];
  const ScalarT& jump = robin_vals[2];

  for(int pt = 0; pt < numPoints; pt++)
    for(int dim = 0; dim < numDOFsSet; dim++)
      qp_data_returned(0, pt, dim) = coeff*(dof_side(0,pt) - dof_value) - jump * scale * 2.0; 
         // mult by 2 to emulate behavior of an internal side within a single material (element block) 
         //  in which case usual Neumann would add contributions from both sides, giving factor of 2
}


template<typename EvalT, typename Traits>
void NeumannBase<EvalT, Traits>::
calc_press(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2); // How many DOFs per node to calculate?

  Intrepid::FieldContainer<MeshScalarT> side_normals(numCells, numPoints, cellDims);
  Intrepid::FieldContainer<MeshScalarT> normal_lengths(numCells, numPoints);
  Intrepid::FieldContainer<MeshScalarT> ref_normal(cellDims);

  // for this side in the reference cell, get the components of the normal direction vector
  Intrepid::CellTools<MeshScalarT>::getPhysicalSideNormals(side_normals, jacobian_side_refcell, 
    local_side_id, celltopo);

  // for this side in the reference cell, get the constant normal vector to the side for area calc
  Intrepid::CellTools<MeshScalarT>::getReferenceSideNormal(ref_normal, local_side_id, celltopo);
  /* Note: if the side is 1D the length of the normal times 2 is the side length
     If the side is a 2D quad, the length of the normal is the area of the side
     If the side is a 2D triangle, the length of the normal times 1/2 is the area of the side
   */

  MeshScalarT area = 
    Intrepid::RealSpaceTools<MeshScalarT>::vectorNorm(ref_normal, Intrepid::NORM_TWO);

  // Calculate proper areas

  switch(side_type){

    case LINE:

      area *= 2;
      break;

    case TRI:

      area /= 2;
      break;

  }

  // scale normals (unity)
  Intrepid::RealSpaceTools<MeshScalarT>::vectorNorm(normal_lengths, side_normals, Intrepid::NORM_TWO);
  Intrepid::FunctionSpaceTools::scalarMultiplyDataData<MeshScalarT>(side_normals, normal_lengths, 
    side_normals, true);

  // Pressure is a force of magnitude P along the normal to the side, divided by the side area (det)

  for(int cell = 0; cell < numCells; cell++)
    for(int pt = 0; pt < numPoints; pt++)
      for(int dim = 0; dim < numDOFsSet; dim++)
//        qp_data_returned(cell, pt, dim) = const_val * side_normals(cell, pt, dim);
        qp_data_returned(cell, pt, dim) = const_val * side_normals(cell, pt, dim) / area;


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

  Teuchos::RCP<Epetra_Vector> f = workset.f;
  ScalarT *valptr;

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  // Place it at the appropriate offset into F
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];


    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);
        (*f)[nodeID[node][this->offset[dim]]] += *valptr;


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
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Epetra_Vector> f = workset.f;
  Teuchos::RCP<Epetra_CrsMatrix> Jac = workset.Jac;
  ScalarT *valptr;

  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  int row, lcol, col;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
     
        valptr = &(this->neumann)(cell, node, dim);

        row = nodeID[node][this->offset[dim]];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) 
          f->SumIntoMyValue(row, 0, valptr->val());

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];
              
              if (workset.is_adjoint) {
                // Sum Jacobian transposed
                Jac->SumIntoMyValues(col, 1, &(valptr->fastAccessDx(lcol)), &row);
              }
              else {
                // Sum Jacobian
                Jac->SumIntoMyValues(row, 1, &(valptr->fastAccessDx(lcol)), &col);
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
  Teuchos::RCP<Epetra_Vector> f = workset.f;
  Teuchos::RCP<Epetra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Epetra_MultiVector> fp = workset.fp;
  ScalarT *valptr;

  const Epetra_BlockMap *row_map = NULL;

  if (f != Teuchos::null)
    row_map = &(f->Map());
  else if (JV != Teuchos::null)
    row_map = &(JV->Map());
  else if (fp != Teuchos::null)
    row_map = &(fp->Map());
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                     "One of f, JV, or fp must be non-null! " << std::endl);

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) 
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (f != Teuchos::null)
          f->SumIntoMyValue(row, 0, valptr->val());

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)

        JV->SumIntoMyValue(row, col, valptr->dx(col));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            fp->SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset));
      }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::SGResidual, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::SGResidual,Traits>(p)
{
}


// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  ScalarT *valptr;

  int nblock = f->size();

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) 
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset[dim]]] += valptr->coeff(block);

    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::SGJacobian, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > Jac = 
    workset.sg_Jac;
  ScalarT *valptr;

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  int row, lcol, col;
  int nblock = 0;

  if (f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        row = nodeID[node][this->offset[dim]];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) {

          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {

                c = valptr->fastAccessDx(lcol).coeff(block);
                if (workset.is_adjoint) { 

                  (*Jac)[block].SumIntoMyValues(col, 1, &c, &row);

                }
                else {

                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);

                }
              }
            } // column equations
          } // column nodes
        } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::SGTangent, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::SGTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > JV = workset.sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > fp = workset.sg_fp;
  ScalarT *valptr;

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
		       "One of sg_f, sg_JV, or sg_fp must be non-null! " << 
		       std::endl);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {

    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::MPResidual, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::MPResidual,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  ScalarT *valptr;

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) 
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][this->offset[dim]]] += valptr->coeff(block);

    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::MPJacobian, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > Jac = 
    workset.mp_Jac;
  ScalarT *valptr;

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  int row, lcol, col;
  int nblock = 0;

  if (f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) 
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        row = nodeID[node][this->offset[dim]];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) 
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));


        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {

                c = valptr->fastAccessDx(lcol).coeff(block);
               (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);

             }
            } // column equations
          } // column nodes
        } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
Neumann<PHAL::AlbanyTraits::MPTangent, Traits>::
Neumann(Teuchos::ParameterList& p)
  : NeumannBase<PHAL::AlbanyTraits::MPTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Neumann<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > JV = workset.mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > fp = workset.mp_fp;
  ScalarT *valptr;

  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
		       "One of mp_f, mp_JV, or mp_fp must be non-null! " << 
		       std::endl);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));

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

  std::vector<std::string>& nbcs = *(p.get<std::vector<std::string>* >("NBC Names"));

  for (unsigned int i=0; i<nbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(nbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("NBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Neumann Aggregator"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
}
