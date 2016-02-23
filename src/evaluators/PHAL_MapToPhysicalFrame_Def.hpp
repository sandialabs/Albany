//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
MapToPhysicalFrame<EvalT, Traits>::
MapToPhysicalFrame(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coords_vertices  (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector),
  cubature         (p.get<Teuchos::RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> > > > ("Intrepid2 Basis") ),
  cellType         (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  coords_qp        (p.get<std::string>  ("Coordinate Vector Name"), dl->qp_gradient)
{
  this->addDependentField(coords_vertices);
  this->addEvaluatedField(coords_qp);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);

  // Compute cubature points in reference elements
  refPoints.resize(dims[1],dims[2]);
  refWeights.resize(dims[1]);
  cubature->getCubature(refPoints, refWeights); 

  this->setName("MapToPhysicalFrame" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords_vertices,fm);
  this->utils.setFieldData(coords_qp,fm);

}
//**********************************************************************
template <class Scalar, class ArrayPhysPoint, class ArrayRefPoint, class ArrayCell>
void mapToPhysicalFrame(ArrayPhysPoint      &        physPoints,
                                           const ArrayRefPoint &        refPoints,
                                           const ArrayCell     &        cellWorkset,
                                           const shards::CellTopology & cellTopo,
                                           const int &                  whichCell)
{
  int spaceDim  = (int)cellTopo.getDimension();
  int numCells  = cellWorkset.dimension(0);
  //points can be rank-2 (P,D), or rank-3 (C,P,D)
  int numPoints = (refPoints.rank() == 2) ? refPoints.dimension(0) : refPoints.dimension(1);

/*  // Mapping is computed using an appropriate H(grad) basis function: define RCP to the base class
  Teuchos::RCP<Basis<Scalar, FieldContainer<Scalar> > > HGRAD_Basis;

  // Choose the H(grad) basis depending on the cell topology. \todo define maps for shells and beams
  switch( cellTopo.getKey() ){

    // Standard Base topologies (number of cellWorkset = number of vertices)
      case shards::Line<2>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_LINE_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Triangle<3>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TRI_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Quadrilateral<4>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Tetrahedron<4>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Hexahedron<8>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Wedge<6>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Pyramid<5>::key:
          HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_PYR_C1_FEM<Scalar, FieldContainer<Scalar> >() );
          break;

    // Standard Extended topologies
    case shards::Triangle<6>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TRI_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Quadrilateral<9>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Tetrahedron<10>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Tetrahedron<11>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_COMP12_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Hexahedron<27>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    case shards::Wedge<18>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;

    // These extended topologies are not used for mapping purposes
    case shards::Quadrilateral<8>::key:
    case shards::Hexahedron<20>::key:
    case shards::Wedge<15>::key:
      TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                          ">>> ERROR (Intrepid2::CellTools::mapToPhysicalFrame): Cell topology not supported. ");
      break;

    // Base and Extended Line, Beam and Shell topologies 
     case shards::Line<3>::key:
    case shards::Beam<2>::key:
    case shards::Beam<3>::key:
    case shards::ShellLine<2>::key:
    case shards::ShellLine<3>::key:
    case shards::ShellTriangle<3>::key:
    case shards::ShellTriangle<6>::key:
    case shards::ShellQuadrilateral<4>::key:
    case shards::ShellQuadrilateral<8>::key:
    case shards::ShellQuadrilateral<9>::key:
      TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                          ">>> ERROR (Intrepid2::CellTools::mapToPhysicalFrame): Cell topology not supported. ");
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                          ">>> ERROR (Intrepid2::CellTools::mapToPhysicalFrame): Cell topology not supported.");
  }// switch  

  // Temp (F,P) array for the values of nodal basis functions at the reference points
  int basisCardinality = HGRAD_Basis -> getCardinality();
  FieldContainer<Scalar> basisVals(basisCardinality, numPoints);
*/
  // Initialize physPoints
  for(int i = 0; i < physPoints.dimentions(0); i++)
    for(int j = 0; j < physPoints.dimentions(1); j++)  
      for(int k = 0; k < physPoints.dimentions(2); k++)
         physPoints(i,j,k) = 0.0;
  

}
//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (intrepidBasis != Teuchos::null){ 
    Intrepid2::CellTools<RealType>::mapToPhysicalFrame
         (coords_qp, refPoints, coords_vertices, intrepidBasis);
  }
  else {
    Intrepid2::CellTools<RealType>::mapToPhysicalFrame
         (coords_qp, refPoints, coords_vertices, *cellType);
  }
 // mapToPhysicalFrame<RealType>(coords_qp, refPoints, coords_vertices, *cellType);
  
}

//**********************************************************************
}

