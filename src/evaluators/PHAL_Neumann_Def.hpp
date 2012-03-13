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

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
Neumann<EvalT, Traits>::
Neumann(const Teuchos::ParameterList& p) :
  sideSetID(p.get<std::string>("Side Set ID")),
  intrepidBasis (p.get<Teuchos::RCP <Intrepid::Basis<RealType, 
                 Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
  coordVec      (p.get<std::string>                   ("Coordinate Vector Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  cubatureCell  (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  cubatureSide  (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Side Cubature")),
  sideType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Side Type")),
  neumann       (p.get<std::string>                   ("Node Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(neumann);

  sideDims = sideType->getDimension();
  numQPsSide = cubatureSide->getNumPoints();

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dim;
  vector_dl->dimensions(dim);
  int containerSize = dim[0];
  numQPs = dim[1];
  cellDims = dim[2];

  // Allocate Temporary FieldContainers
  cubPointsSide.resize(numQPsSide, sideDims);
  refPointsSide.resize(numQPsSide, cellDims);
  cubWeightsSide.resize(numQPsSide);
  physPointsSide.resize(1, numQPsSide, cellDims);

  // Do the BC one side at a time for now
  jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
  jacobianSide_det.resize(1, numQPsSide);

  weighted_measure.resize(1, numQPsSide);
  basis_refPointsSide.resize(numNodes, numQPsSide);
  trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
  weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);

  data.resize(1, numQPsSide);

  // Pre-Calculate reference element quantitites
  cubatureSide->getCubature(cubPointsSide, cubWeightsSide);

  this->setName("Neumann"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Neumann<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(neumann, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Neumann<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from 
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->sideSetID);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
   for (std::size_t node=0; node < numNodes; ++node) {          
	   neumann(cell, node) = 0.0;
   }
  }

  if(it == ssList.end()) return; // This sideset does not exist in this workset (GAH - this can go away
                                  // once we move logic to BCUtils

  const std::vector<Albany::SideStruct>& sideSet = it->second;

  // Loop over the sides that form the boundary condition 

  for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

    // Get the data that corresponds to the side

    const int elem_GID = sideSet[side].elem_GID;
    const int elem_LID = sideSet[side].elem_LID;
    const int elem_side = sideSet[side].side_local_id;

    // Map side cubature points to the reference parent cell based on the appropriate side (elem_side) 
    Intrepid::CellTools<RealType>::mapToReferenceSubcell
      (refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

    // Calculate side geometry
    Intrepid::CellTools<RealType>::setJacobian
       (jacobianSide, refPointsSide, coordVec, *cellType, elem_LID);
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
      (physPointsSide, refPointsSide, coordVec, *cellType, elem_LID);

    // Transform the given BC data to the physical space QPs in each side (elem_side)
//   calc_gradT_dotn_const(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side);
   calc_dTdn_const(data, physPointsSide, jacobianSide, *cellType, cellDims, elem_side);

   // Put this side's contribution into the vector

   for (std::size_t node=0; node < numNodes; ++node) {          
     for (std::size_t qp=0; qp < numQPsSide; ++qp) {               
	     neumann(elem_LID, node) += 
         data(0, qp) * weighted_trans_basis_refPointsSide(0, node, qp);

     }
   }
  }
  
}

template<typename EvalT, typename Traits>
void Neumann<EvalT, Traits>::
calc_gradT_dotn_const(Intrepid::FieldContainer<typename EvalT::MeshScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<typename EvalT::MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<typename EvalT::MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?

  Intrepid::FieldContainer<RealType> grad_T(numCells, numPoints, cellDims);
  Intrepid::FieldContainer<typename EvalT::MeshScalarT> side_normals(numCells, numPoints, cellDims);
  Intrepid::FieldContainer<typename EvalT::MeshScalarT> normal_lengths(numCells, numPoints);

  double kdTdx[3];
  kdTdx[0] = 1.0; // Neumann component in the x direction
  kdTdx[1] = 0.0; // Neumann component in the y direction
  kdTdx[2] = 0.0; // Neumann component in the z direction

  for(int cell = 0; cell < numCells; cell++)

    for(int pt = 0; pt < numPoints; pt++){

      for(int dim = 0; dim < cellDims; dim++){

        grad_T(cell, pt, dim) = kdTdx[dim]; // k grad T in the x direction goes in the x spot, and so on

      }
  }

  // for this side in the reference cell, get the components of the normal direction vector
  Intrepid::CellTools<typename EvalT::MeshScalarT>::getPhysicalSideNormals(side_normals, jacobian_side_refcell, 
    local_side_id, celltopo);

  // scale normals (unity)
  Intrepid::RealSpaceTools<typename EvalT::MeshScalarT>::vectorNorm(normal_lengths, side_normals, Intrepid::NORM_TWO);
  Intrepid::FunctionSpaceTools::scalarMultiplyDataData<typename EvalT::MeshScalarT>(side_normals, normal_lengths, 
    side_normals, true);

  // take grad_T dotted with the unit normal
  Intrepid::FunctionSpaceTools::dotMultiplyDataData<typename EvalT::MeshScalarT>(qp_data_returned, 
    grad_T, side_normals);

}

template<typename EvalT, typename Traits>
void Neumann<EvalT, Traits>::
calc_dTdn_const(Intrepid::FieldContainer<typename EvalT::MeshScalarT> & qp_data_returned,
                          const Intrepid::FieldContainer<typename EvalT::MeshScalarT>& phys_side_cub_points,
                          const Intrepid::FieldContainer<typename EvalT::MeshScalarT>& jacobian_side_refcell,
                          const shards::CellTopology & celltopo,
                          const int cellDims,
                          int local_side_id){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?

  double input_dTdn = -1.0; // input number (negative for flux in opposite normal direction)

  for(int pt = 0; pt < numPoints; pt++){

    qp_data_returned(0, pt) = input_dTdn; // User directly specified dTdn, just use it

  }

}


// **********************************************************************
// Simple evaluator to aggregate all Neumann BCs into one "field"
// **********************************************************************

template<typename EvalT, typename Traits>
NeumannAggregator<EvalT, Traits>::
NeumannAggregator(Teuchos::ParameterList& p) 
{
  Teuchos::RCP<PHX::DataLayout> dl =  p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  std::vector<std::string>& nbcs = *(p.get<std::vector<std::string>* >("NBC Names"));

  for (unsigned int i=0; i<nbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(nbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("NBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Nirichlet Aggregator"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
}
