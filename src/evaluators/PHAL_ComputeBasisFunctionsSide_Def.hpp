//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace PHAL {

template<typename EvalT, typename Traits>
ComputeBasisFunctionsSide<EvalT, Traits>::
ComputeBasisFunctionsSide (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl_side) :
  sideCoordVec      (p.get<std::string> ("Side Coordinate Vector Name"), dl_side->vertices_vector ),
  inv_metric    (p.get<std::string> ("Inverse Metric Name"), dl_side->qp_tensor ),
  w_measure     (p.get<std::string> ("Weighted Measure Name"), dl_side->qp_scalar ),
  metric_det    (p.get<std::string> ("Metric Determinant Name"), dl_side->qp_scalar ),
  BF            (p.get<std::string> ("BF Name"), dl_side->node_qp_scalar),
  GradBF        (p.get<std::string> ("Gradient BF Name"), dl_side->node_qp_gradient)
{
  this->addDependentField(sideCoordVec);
  this->addEvaluatedField(w_measure);
  this->addEvaluatedField(metric_det);
  this->addEvaluatedField(inv_metric);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(GradBF);


  compute_side_normals = p.isParameter("Side Normals Name");
  if(compute_side_normals) {
    side_normals = decltype(side_normals)(p.get<std::string> ("Side Normals Name"), dl_side->qp_gradient);
    Teuchos::RCP<Albany::Layouts> dl = p.get<Teuchos::RCP<Albany::Layouts>>("Layout Name");
    coordVec = decltype(coordVec)(p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector );
    numNodes = dl->node_gradient->dimension(1);
    this->addEvaluatedField(side_normals);
    this->addEvaluatedField(coordVec);
  }

  sideSetName = p.get<std::string>("Side Set Name");


  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  int numCells = dl_side->node_qp_gradient->dimension(0);
  numSides     = dl_side->node_qp_gradient->dimension(1);
  numSideNodes = dl_side->node_qp_gradient->dimension(2);
  numSideQPs   = dl_side->node_qp_gradient->dimension(3);
  cellDims     = dl_side->node_qp_gradient->dimension(4);
  sideDims     = cellDims-1;

  cubature = p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature Side");
  intrepidBasis = p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid Basis Side");

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  *output << "Compute Basis Functions Side has: "
          << numCells << " cells, "
          << numSides << " sides, "
          << numSideNodes << " side nodes, "
          << numSideQPs << " side QPs, "
          << sideDims << " side dimensions.\n";
#endif

  this->setName("ComputeBasisFunctionsSide"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctionsSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sideCoordVec,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(metric_det,fm);
  this->utils.setFieldData(inv_metric,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);

  if(compute_side_normals) {
    this->utils.setFieldData(side_normals,fm);
    this->utils.setFieldData(coordVec, fm);
  }

  tangents = Kokkos::createDynRankView(metric_det.get_view(), "XXX", sideDims,cellDims,numSideQPs);
  metric = Kokkos::createDynRankView(metric_det.get_view(), "XXX", numSideQPs,sideDims,sideDims);

  // Allocate Temporary Kokkos Views
  cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs,sideDims);
  cub_weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs);
  val_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideNodes, numSideQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideNodes, numSideQPs, sideDims);
  cub_weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(cub_points, cub_weights);

  intrepidBasis->getValues(val_at_cub_points, cub_points, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, cub_points, Intrepid2::OPERATOR_GRAD);

  // BF does not depend on the current element, so we fill it now
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  int numCells = dims[0];
  for(int cell=0; cell<numCells; ++cell)
  {
    for (int side=0; side<numSides; ++side)
    {
      for (int node=0; node<numSideNodes; ++node)
      {
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          BF(cell,side,node,qp) = val_at_cub_points(node,qp);
        }
      }
    }
  }

  cellsOnSides.resize(numSides);
  numCellsOnSide.resize(numSides, 0);
  for (int i=0; i<numSides; i++)
    cellsOnSides[i] = Kokkos::DynRankView<int, PHX::Device>("cellOnSide_i", numCells);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctionsSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  //TODO: use Intrepid routines as much as possible
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  numCellsOnSide.assign(numSides, 0);
  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    cellsOnSides[side](numCellsOnSide[side]++) = cell;

    // Computing tangents (the basis for the manifold)
    for (int itan=0; itan<sideDims; ++itan)
    {
      for (int icoor=0; icoor<cellDims; ++icoor)
      {
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          tangents(itan,icoor,qp) = 0.;
          for (int node=0; node<numSideNodes; ++node)
          {
            tangents(itan,icoor,qp) += sideCoordVec(cell,side,node,icoor) * grad_at_cub_points(node,qp,itan);
          }
        }
      }
    }
    // Computing the metric
    for (int qp=0; qp<numSideQPs; ++qp)
    {
      for (int idim=0; idim<sideDims; ++idim)
      {
        // Diagonal
        metric(qp,idim,idim) = 0.;
        for (int coor=0; coor<cellDims; ++coor)
        {
          metric(qp,idim,idim) += tangents(idim,coor,qp)*tangents(idim,coor,qp); // g = J'*J
        }

        // Extra-diagonal
        for (int jdim=idim+1; jdim<sideDims; ++jdim)
        {
          metric(qp,idim,jdim) = 0.;
          for (int coor=0; coor<cellDims; ++coor)
          {
            metric(qp,idim,jdim) += tangents(idim,coor,qp)*tangents(jdim,coor,qp); // g = J'*J
          }
          metric(qp,jdim,idim) = metric(qp,idim,jdim);
        }
      }
    }

    // Computing the metric determinant, the weighted measure and the inverse of the metric
    switch (sideDims)
    {
      case 1:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          metric_det(cell,side,qp) = metric(qp,0,0);
          w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt(metric(qp,0,0));
          inv_metric(cell,side,qp,0,0) = 1./metric(qp,0,0);
        }
        break;
      case 2:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          metric_det(cell,side,qp) = metric(qp,0,0)*metric(qp,1,1) - metric(qp,0,1)*metric(qp,1,0);
          w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt(metric_det(cell,side,qp));
          inv_metric(cell,side,qp,0,0) = metric(qp,1,1)/metric_det(cell,side,qp);
          inv_metric(cell,side,qp,1,1) = metric(qp,0,0)/metric_det(cell,side,qp);
          inv_metric(cell,side,qp,0,1) = inv_metric(cell,side,qp,1,0) = -metric(qp,0,1)/metric_det(cell,side,qp);
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The dimension of the side should be 1 or 2.\n");
    }

    for (int node=0; node<numSideNodes; ++node)
    {
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        for (int ider=0; ider<cellDims; ++ider)
        {
          GradBF(cell,side,node,qp,ider)=0;
          for(int j=0; j< sideDims; ++j)
            for(int k=0; k< sideDims; ++k)
              GradBF(cell,side,node,qp,ider) +=  tangents(j,ider,qp)*inv_metric(cell,side,qp,j,k)*grad_at_cub_points(node,qp,k);
        }
      }
    }
  }

  if(compute_side_normals){
    for (int side = 0; side < numSides; ++side)
    {          
      int numCells_ =  numCellsOnSide[side];
      if( numCells_ == 0) continue;
      
      Kokkos::DynRankView<MeshScalarT, PHX::Device> normal_lengths = Kokkos::createDynRankView(sideCoordVec.get_view(),"normal_lengths", numCells_, numSideQPs);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> normals = Kokkos::createDynRankView(sideCoordVec.get_view(),"normals", numCells_, numSideQPs, cellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobian_side = Kokkos::createDynRankView(sideCoordVec.get_view(),"jacobian_side", numCells_, numSideQPs, cellDims, cellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsSide = Kokkos::createDynRankView(sideCoordVec.get_view(),"physPointsSide", numCells_, numSideQPs, cellDims);
      Kokkos::DynRankView<RealType, PHX::Device> refPointsSide("refPointsSide", numSideQPs, cellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell = Kokkos::createDynRankView(coordVec.get_view(), "XXX", numCells_, numNodes, cellDims);
      Kokkos::DynRankView<int, PHX::Device> cellVec  = cellsOnSides[side];
      
      

      for (std::size_t node=0; node < numNodes; ++node)
        for (std::size_t dim=0; dim < cellDims; ++dim)
          for (std::size_t iCell=0; iCell < numCells_; ++iCell)
            physPointsCell(iCell, node, dim) = coordVec(cellVec(iCell),node,dim);

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell
        (refPointsSide, cub_points, sideDims, side, *cellType);

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian
       (jacobian_side, refPointsSide, physPointsCell, *cellType);


      // for this side in the reference cell, get the components of the normal direction vector
      Intrepid2::CellTools<PHX::Device>::getPhysicalSideNormals(normals, jacobian_side, side, *cellType);

      // scale normals (unity)
      Intrepid2::RealSpaceTools<PHX::Device>::vectorNorm(normal_lengths, normals, Intrepid2::NORM_TWO);
      Intrepid2::FunctionSpaceTools<PHX::Device>::scalarMultiplyDataData(normals, normal_lengths, normals, true);

      for (int icoor=0; icoor<cellDims; ++icoor)
        for (int qp=0; qp<numSideQPs; ++qp)
          for (std::size_t iCell=0; iCell < numCells_; ++iCell)
            side_normals(cellVec(iCell),side,qp, icoor) = normals(iCell,qp,icoor);
    }
  }
}

} // Namespace PHAL
