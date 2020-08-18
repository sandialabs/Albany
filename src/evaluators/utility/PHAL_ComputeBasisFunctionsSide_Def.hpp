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
                           const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Get side set name and side set layouts
  sideSetName = p.get<std::string>("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(),
                              std::runtime_error, "Error! Layouts for side set '" << sideSetName << "' not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  // Build output fields
  sideCoordVec = decltype(sideCoordVec)(p.get<std::string> ("Side Coordinate Vector Name"), (dl_side->useCollapsedSidesets) ? dl_side->vertices_vector_sideset : dl_side->vertices_vector );
  tangents     = decltype(tangents    )(p.get<std::string> ("Tangents Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_tensor_cd_sd_sideset : dl_side->qp_tensor_cd_sd );
  metric       = decltype(metric      )(p.get<std::string> ("Metric Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_tensor_sideset : dl_side->qp_tensor );
  w_measure    = decltype(w_measure   )(p.get<std::string> ("Weighted Measure Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_scalar_sideset : dl_side->qp_scalar );
  inv_metric   = decltype(inv_metric  )(p.get<std::string> ("Inverse Metric Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_tensor_sideset : dl_side->qp_tensor );
  metric_det   = decltype(metric_det  )(p.get<std::string> ("Metric Determinant Name"), (dl_side->useCollapsedSidesets) ? dl_side->qp_scalar_sideset : dl_side->qp_scalar );
  BF           = decltype(BF          )(p.get<std::string> ("BF Name"), (dl_side->useCollapsedSidesets) ? dl_side->node_qp_scalar_sideset : dl_side->node_qp_scalar);
  GradBF       = decltype(GradBF      )(p.get<std::string> ("Gradient BF Name"), (dl_side->useCollapsedSidesets) ? dl_side->node_qp_gradient_sideset : dl_side->node_qp_gradient);

  useCollapsedSidesets = dl_side->useCollapsedSidesets;

  this->addDependentField(sideCoordVec);
  this->addEvaluatedField(tangents);
  this->addEvaluatedField(metric);
  this->addEvaluatedField(metric_det);
  this->addEvaluatedField(w_measure);
  this->addEvaluatedField(inv_metric);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(GradBF);

  compute_normals = p.isParameter("Side Normal Name");

  if(compute_normals) {
    normals  = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_side->qp_vector_spacedim);
    coordVec = decltype(coordVec)(p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector );
    numNodes = dl->node_gradient->extent(1);
    this->addEvaluatedField(normals);
    this->addDependentField(coordVec);
  }

  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells     = dl_side->node_qp_gradient->extent(0);
  numSides     = dl_side->node_qp_gradient->extent(1);
  numSideNodes = dl_side->node_qp_gradient->extent(2);
  numSideQPs   = dl_side->node_qp_gradient->extent(3);
  numCellDims  = dl_side->vertices_vector->extent(3);    // Vertices vector always has the ambient space dimension
  numSideDims  = numCellDims-1;

  effectiveCoordDim = p.get<bool>("Side Set Is Planar") ? 2 : numCellDims;

  cubature = p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature Side");
  intrepidBasis = p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid Basis Side");

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  *output << "Compute Basis Functions Side has: "
          << numCells << " cells, "
          << numSides << " sides, "
          << numSideNodes << " side nodes, "
          << numSideQPs << " side QPs, "
          << numSideDims << " side dimensions.\n";
#endif

  this->setName("ComputeBasisFunctionsSide"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctionsSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sideCoordVec,fm);
  this->utils.setFieldData(tangents,fm);
  this->utils.setFieldData(metric,fm);
  this->utils.setFieldData(metric_det,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(inv_metric,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);

  if(compute_normals) {
    this->utils.setFieldData(normals,fm);
    this->utils.setFieldData(coordVec, fm);
  }

  // Allocate Temporary Kokkos Views
  cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs,numSideDims);
  cub_weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs);
  val_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideNodes, numSideQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideNodes, numSideQPs, numSideDims);
  cub_weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(cub_points, cub_weights);

  intrepidBasis->getValues(val_at_cub_points, cub_points, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, cub_points, Intrepid2::OPERATOR_GRAD);

  // BF does not depend on the current element, so we fill it now
  if (!useCollapsedSidesets) {
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
  } else {
    for (int cellside = 0; cellside < BF.extent(0); ++cellside)
    {
      for (int node=0; node<numSideNodes; ++node)
      {
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          BF(cellside,node,qp) = val_at_cub_points(node,qp);
        }
      }
    }
  }

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctionsSide<EvalT, Traits>::
operator() (const ComputeBasisFunctionsSide_Collapsed_Tag& tag, const int& sideSet_idx) const {

  // Get the local data of side and cell
  const int cell = sideSet.elem_LID(sideSet_idx);
  const int side = sideSet.side_local_id(sideSet_idx);

  // Computing tangents (the basis for the manifold)
  for (int itan=0; itan<numSideDims; ++itan) {
    for (int icoor=0; icoor<numCellDims; ++icoor) {
      for (int qp=0; qp<numSideQPs; ++qp) {
        tangents(sideSet_idx,qp,icoor,itan) = 0.;
        if(icoor < effectiveCoordDim)   //if it's planar do not compute the z dimension
        for (int node=0; node<numSideNodes; ++node) {
          tangents(sideSet_idx,qp,icoor,itan) += sideCoordVec(sideSet_idx,node,icoor) * grad_at_cub_points(node,qp,itan);
        }
      }
    }
  }

  // Computing the metric
  for (int qp=0; qp<numSideQPs; ++qp) {
    for (int idim=0; idim<numSideDims; ++idim) {
      // Diagonal
      metric(sideSet_idx,qp,idim,idim) = 0.;
      for (int coor=0; coor<numCellDims; ++coor) {
        metric(sideSet_idx,qp,idim,idim) += tangents(sideSet_idx,qp,coor,idim)*tangents(sideSet_idx,qp,coor,idim); // g = J'*J
      }

      // Extra-diagonal
      for (int jdim=idim+1; jdim<numSideDims; ++jdim) {
        metric(sideSet_idx,qp,idim,jdim) = 0.;
        for (int coor=0; coor<numCellDims; ++coor) {
          metric(sideSet_idx,qp,idim,jdim) += tangents(sideSet_idx,qp,coor,idim)*tangents(sideSet_idx,qp,coor,jdim); // g = J'*J
        }
        metric(sideSet_idx,qp,jdim,idim) =  metric(sideSet_idx,qp,idim,jdim);
      }
    }
  }

  // Computing the metric determinant, the weighted measure and the inverse of the metric
  switch (numSideDims) {
    case 1:
      for (int qp=0; qp<numSideQPs; ++qp) {
        metric_det(sideSet_idx,qp) =  metric(sideSet_idx,qp,0,0);
        w_measure(sideSet_idx,qp) = cub_weights(qp)*std::sqrt( metric(sideSet_idx,qp,0,0));
        inv_metric(sideSet_idx,qp,0,0) = 1./ metric(sideSet_idx,qp,0,0);
      }
      break;
    case 2:
      for (int qp=0; qp<numSideQPs; ++qp) {
        metric_det(sideSet_idx,qp) =  metric(sideSet_idx,qp,0,0)* metric(sideSet_idx,qp,1,1) -  metric(sideSet_idx,qp,0,1)* metric(sideSet_idx,qp,1,0);
        w_measure(sideSet_idx,qp) = cub_weights(qp)*std::sqrt(metric_det(sideSet_idx,qp));
        inv_metric(sideSet_idx,qp,0,0) =  metric(sideSet_idx,qp,1,1)/metric_det(sideSet_idx,qp);
        inv_metric(sideSet_idx,qp,1,1) =  metric(sideSet_idx,qp,0,0)/metric_det(sideSet_idx,qp);
        inv_metric(sideSet_idx,qp,0,1) = inv_metric(sideSet_idx,qp,1,0) = - metric(sideSet_idx,qp,0,1)/metric_det(sideSet_idx,qp); 
      }
      break;
    default:
      break;
      // TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The dimension of the side should be 1 or 2.\n");
  }

  for (int node=0; node<numSideNodes; ++node) {
    for (int qp=0; qp<numSideQPs; ++qp) {
      for (int ider=0; ider<numSideDims; ++ider) {
        GradBF(sideSet_idx,node,qp,ider)=0;
        for(int jder=0; jder< numSideDims; ++jder) {
          GradBF(sideSet_idx,node,qp,ider) += inv_metric(sideSet_idx,qp,ider,jder)*grad_at_cub_points(node,qp,jder);
        }
      }
    }
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctionsSide<EvalT, Traits>::
operator() (const ComputeBasisFunctionsSide_Tag& tag, const int& sideSet_idx) const {

  // Get the local data of side and cell
  const int cell = sideSet.elem_LID(sideSet_idx);
  const int side = sideSet.side_local_id(sideSet_idx);

  // Computing tangents (the basis for the manifold)
  for (int itan=0; itan<numSideDims; ++itan) {
    for (int icoor=0; icoor<numCellDims; ++icoor) {
      for (int qp=0; qp<numSideQPs; ++qp) {
        tangents(cell,side,qp,icoor,itan) = 0.;
        if(icoor < effectiveCoordDim)   //if it's planar do not compute the z dimension
        for (int node=0; node<numSideNodes; ++node) {
          tangents(cell,side,qp,icoor,itan) += sideCoordVec(cell,side,node,icoor) * grad_at_cub_points(node,qp,itan);
        }
      }
    }
  }

  // Computing the metric
  for (int qp=0; qp<numSideQPs; ++qp) {
    for (int idim=0; idim<numSideDims; ++idim) {
      // Diagonal
      metric(cell,side,qp,idim,idim) = 0.;
      for (int coor=0; coor<numCellDims; ++coor) {
        metric(cell,side,qp,idim,idim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,idim); // g = J'*J
      }

      // Extra-diagonal
      for (int jdim=idim+1; jdim<numSideDims; ++jdim) {
        metric(cell,side,qp,idim,jdim) = 0.;
        for (int coor=0; coor<numCellDims; ++coor) {
          metric(cell,side,qp,idim,jdim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,jdim); // g = J'*J
        }
        metric(cell,side,qp,jdim,idim) =  metric(cell,side,qp,idim,jdim);
      }
    }
  }

  // Computing the metric determinant, the weighted measure and the inverse of the metric
  switch (numSideDims) {
    case 1:
      for (int qp=0; qp<numSideQPs; ++qp) {
        metric_det(cell,side,qp) =  metric(cell,side,qp,0,0);
        w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt( metric(cell,side,qp,0,0));
        inv_metric(cell,side,qp,0,0) = 1./ metric(cell,side,qp,0,0);
      }
      break;
    case 2:
      for (int qp=0; qp<numSideQPs; ++qp) {
        metric_det(cell,side,qp) =  metric(cell,side,qp,0,0)* metric(cell,side,qp,1,1) -  metric(cell,side,qp,0,1)* metric(cell,side,qp,1,0);
        w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt(metric_det(cell,side,qp));
        inv_metric(cell,side,qp,0,0) =  metric(cell,side,qp,1,1)/metric_det(cell,side,qp);
        inv_metric(cell,side,qp,1,1) =  metric(cell,side,qp,0,0)/metric_det(cell,side,qp);
        inv_metric(cell,side,qp,0,1) = inv_metric(cell,side,qp,1,0) = - metric(cell,side,qp,0,1)/metric_det(cell,side,qp); 
      }
      break;
    default:
      break;
  }

  for (int node=0; node<numSideNodes; ++node) {
    for (int qp=0; qp<numSideQPs; ++qp) {
      for (int ider=0; ider<numSideDims; ++ider) {
        GradBF(cell,side,node,qp,ider)=0;
        for(int jder=0; jder< numSideDims; ++jder) {
          GradBF(cell,side,node,qp,ider) += inv_metric(cell,side,qp,ider,jder)*grad_at_cub_points(node,qp,jder);
        }
      }
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctionsSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  //TODO: use Intrepid routines as much as possible
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return;

  sideSet = workset.sideSetViews->at(sideSetName);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (useCollapsedSidesets) {
    Kokkos::parallel_for(ComputeBasisFunctionsSide_Collapsed_Policy(0,sideSet.size), *this);
  } else {
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.elem_LID(sideSet_idx);
      const int side = sideSet.side_local_id(sideSet_idx);

      // Computing tangents (the basis for the manifold)
      for (int itan=0; itan<numSideDims; ++itan) {
        for (int icoor=0; icoor<numCellDims; ++icoor) {
          for (int qp=0; qp<numSideQPs; ++qp) {
            tangents(cell,side,qp,icoor,itan) = 0.;
            if(icoor < effectiveCoordDim)   //if it's planar do not compute the z dimension
            for (int node=0; node<numSideNodes; ++node) {
              tangents(cell,side,qp,icoor,itan) += sideCoordVec(cell,side,node,icoor) * grad_at_cub_points(node,qp,itan);
            }
          }
        }
      }

      // Computing the metric
      for (int qp=0; qp<numSideQPs; ++qp) {
        for (int idim=0; idim<numSideDims; ++idim) {
          // Diagonal
          metric(cell,side,qp,idim,idim) = 0.;
          for (int coor=0; coor<numCellDims; ++coor) {
            metric(cell,side,qp,idim,idim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,idim); // g = J'*J
          }

          // Extra-diagonal
          for (int jdim=idim+1; jdim<numSideDims; ++jdim) {
            metric(cell,side,qp,idim,jdim) = 0.;
            for (int coor=0; coor<numCellDims; ++coor) {
              metric(cell,side,qp,idim,jdim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,jdim); // g = J'*J
            }
            metric(cell,side,qp,jdim,idim) =  metric(cell,side,qp,idim,jdim);
          }
        }
      }

      // Computing the metric determinant, the weighted measure and the inverse of the metric
      switch (numSideDims) {
        case 1:
          for (int qp=0; qp<numSideQPs; ++qp) {
            metric_det(cell,side,qp) =  metric(cell,side,qp,0,0);
            w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt( metric(cell,side,qp,0,0));
            inv_metric(cell,side,qp,0,0) = 1./ metric(cell,side,qp,0,0);
          }
          break;
        case 2:
          for (int qp=0; qp<numSideQPs; ++qp) {
            metric_det(cell,side,qp) =  metric(cell,side,qp,0,0)* metric(cell,side,qp,1,1) -  metric(cell,side,qp,0,1)* metric(cell,side,qp,1,0);
            w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt(metric_det(cell,side,qp));
            inv_metric(cell,side,qp,0,0) =  metric(cell,side,qp,1,1)/metric_det(cell,side,qp);
            inv_metric(cell,side,qp,1,1) =  metric(cell,side,qp,0,0)/metric_det(cell,side,qp);
            inv_metric(cell,side,qp,0,1) = inv_metric(cell,side,qp,1,0) = - metric(cell,side,qp,0,1)/metric_det(cell,side,qp); 
          }
          break;
        default:
          break;
      }

      for (int node=0; node<numSideNodes; ++node) {
        for (int qp=0; qp<numSideQPs; ++qp) {
          for (int ider=0; ider<numSideDims; ++ider) {
            GradBF(cell,side,node,qp,ider)=0;
            for(int jder=0; jder< numSideDims; ++jder) {
              GradBF(cell,side,node,qp,ider) += inv_metric(cell,side,qp,ider,jder)*grad_at_cub_points(node,qp,jder);
            }
          }
        }
      }
    }

  }
#else
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    // Get the local data of side and cell
    const int cell = sideSet.elem_LID(sideSet_idx);
    const int side = sideSet.side_local_id(sideSet_idx);

    // Computing tangents (the basis for the manifold)
    for (int itan=0; itan<numSideDims; ++itan)
    {
      for (int icoor=0; icoor<numCellDims; ++icoor)
      {
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          if (useCollapsedSidesets) {
            tangents(sideSet_idx,qp,icoor,itan) = 0.;
            if(icoor < effectiveCoordDim)   //if it's planar do not compute the z dimension
            for (int node=0; node<numSideNodes; ++node) {
              tangents(sideSet_idx,qp,icoor,itan) += sideCoordVec(sideSet_idx,node,icoor) * grad_at_cub_points(node,qp,itan);
            }
          } else {
            tangents(cell,side,qp,icoor,itan) = 0.;
            if(icoor < effectiveCoordDim)   //if it's planar do not compute the z dimension
            for (int node=0; node<numSideNodes; ++node) {
              tangents(cell,side,qp,icoor,itan) += sideCoordVec(cell,side,node,icoor) * grad_at_cub_points(node,qp,itan);
            }
          }
        }
      }
    }
    // Computing the metric
    for (int qp=0; qp<numSideQPs; ++qp)
    {
      for (int idim=0; idim<numSideDims; ++idim)
      {
        if (useCollapsedSidesets) {
          // Diagonal
          metric(sideSet_idx,qp,idim,idim) = 0.;
          for (int coor=0; coor<numCellDims; ++coor)
          {
            metric(sideSet_idx,qp,idim,idim) += tangents(sideSet_idx,qp,coor,idim)*tangents(sideSet_idx,qp,coor,idim); // g = J'*J
          }

          // Extra-diagonal
          for (int jdim=idim+1; jdim<numSideDims; ++jdim)
          {
            metric(sideSet_idx,qp,idim,jdim) = 0.;
            for (int coor=0; coor<numCellDims; ++coor)
            {
              metric(sideSet_idx,qp,idim,jdim) += tangents(sideSet_idx,qp,coor,idim)*tangents(sideSet_idx,qp,coor,jdim); // g = J'*J
            }
            metric(sideSet_idx,qp,jdim,idim) =  metric(sideSet_idx,qp,idim,jdim);
          }
        } else {
          // Diagonal
          metric(cell,side,qp,idim,idim) = 0.;
          for (int coor=0; coor<numCellDims; ++coor)
          {
            metric(cell,side,qp,idim,idim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,idim); // g = J'*J
          }

          // Extra-diagonal
          for (int jdim=idim+1; jdim<numSideDims; ++jdim)
          {
            metric(cell,side,qp,idim,jdim) = 0.;
            for (int coor=0; coor<numCellDims; ++coor)
            {
              metric(cell,side,qp,idim,jdim) += tangents(cell,side,qp,coor,idim)*tangents(cell,side,qp,coor,jdim); // g = J'*J
            }
            metric(cell,side,qp,jdim,idim) =  metric(cell,side,qp,idim,jdim);
          }
        }
      }
    }

    // Computing the metric determinant, the weighted measure and the inverse of the metric
    switch (numSideDims)
    {
      case 1:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          if (useCollapsedSidesets) {
            metric_det(sideSet_idx,qp) =  metric(sideSet_idx,qp,0,0);
            w_measure(sideSet_idx,qp) = cub_weights(qp)*std::sqrt( metric(sideSet_idx,qp,0,0));
            inv_metric(sideSet_idx,qp,0,0) = 1./ metric(sideSet_idx,qp,0,0);
          } else {
            metric_det(cell,side,qp) =  metric(cell,side,qp,0,0);
            w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt( metric(cell,side,qp,0,0));
            inv_metric(cell,side,qp,0,0) = 1./ metric(cell,side,qp,0,0);
          }
        }
        break;
      case 2:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          if (useCollapsedSidesets) {
            metric_det(sideSet_idx,qp) =  metric(sideSet_idx,qp,0,0)* metric(sideSet_idx,qp,1,1) -  metric(sideSet_idx,qp,0,1)* metric(sideSet_idx,qp,1,0);
            w_measure(sideSet_idx,qp) = cub_weights(qp)*std::sqrt(metric_det(sideSet_idx,qp));
            inv_metric(sideSet_idx,qp,0,0) =  metric(sideSet_idx,qp,1,1)/metric_det(sideSet_idx,qp);
            inv_metric(sideSet_idx,qp,1,1) =  metric(sideSet_idx,qp,0,0)/metric_det(sideSet_idx,qp);
            inv_metric(sideSet_idx,qp,0,1) = inv_metric(sideSet_idx,qp,1,0) = - metric(sideSet_idx,qp,0,1)/metric_det(sideSet_idx,qp); 
          } else {
            metric_det(cell,side,qp) =  metric(cell,side,qp,0,0)* metric(cell,side,qp,1,1) -  metric(cell,side,qp,0,1)* metric(cell,side,qp,1,0);
            w_measure(cell,side,qp) = cub_weights(qp)*std::sqrt(metric_det(cell,side,qp));
            inv_metric(cell,side,qp,0,0) =  metric(cell,side,qp,1,1)/metric_det(cell,side,qp);
            inv_metric(cell,side,qp,1,1) =  metric(cell,side,qp,0,0)/metric_det(cell,side,qp);
            inv_metric(cell,side,qp,0,1) = inv_metric(cell,side,qp,1,0) = - metric(cell,side,qp,0,1)/metric_det(cell,side,qp); 
          }
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The dimension of the side should be 1 or 2.\n");
    }

    for (int node=0; node<numSideNodes; ++node)
    {
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        for (int ider=0; ider<numSideDims; ++ider)
        {
          if (useCollapsedSidesets) {
            GradBF(sideSet_idx,node,qp,ider)=0;
            for(int jder=0; jder< numSideDims; ++jder)
              GradBF(sideSet_idx,node,qp,ider) += inv_metric(sideSet_idx,qp,ider,jder)*grad_at_cub_points(node,qp,jder);
          } else {
            GradBF(sideSet_idx,node,qp,ider)=0;
            for(int jder=0; jder< numSideDims; ++jder)
              GradBF(sideSet_idx,node,qp,ider) += inv_metric(sideSet_idx,qp,ider,jder)*grad_at_cub_points(node,qp,jder);
          }
        }
      }
    }
  }
#endif

  if(compute_normals){
    int numSides_ = sideSet.numCellsOnSide.extent(0);
    for (int side = 0; side < numSides_; ++side)
    {
      int numCells_ = sideSet.numCellsOnSide(side);
      if( numCells_ == 0) continue;

      Kokkos::DynRankView<MeshScalarT, PHX::Device> normal_lengths = Kokkos::createDynRankView(sideCoordVec.get_view(),"normal_lengths", numCells_, numSideQPs);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> normals_view = Kokkos::createDynRankView(sideCoordVec.get_view(),"normals", numCells_, numSideQPs, numCellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobian_side = Kokkos::createDynRankView(sideCoordVec.get_view(),"jacobian_side", numCells_, numSideQPs, numCellDims, numCellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsSide = Kokkos::createDynRankView(sideCoordVec.get_view(),"physPointsSide", numCells_, numSideQPs, numCellDims);
      Kokkos::DynRankView<RealType, PHX::Device> refPointsSide("refPointsSide", numSideQPs, numCellDims);
      Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell = Kokkos::createDynRankView(coordVec.get_view(), "XXX", numCells_, numNodes, numCellDims);

      for (std::size_t iCell=0; iCell < numCells_; ++iCell)
        for (std::size_t node=0; node < numNodes; ++node)
          for (std::size_t dim=0; dim < numCellDims; ++dim)
            physPointsCell(iCell, node, dim) = coordVec(sideSet.cellsOnSide(side,iCell),node,dim);

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell
        (refPointsSide, cub_points, numSideDims, side, *cellType);

      // Calculate side geometry
      Intrepid2::CellTools<PHX::Device>::setJacobian
       (jacobian_side, refPointsSide, physPointsCell, *cellType);


      // for this side in the reference cell, get the components of the normal direction vector
      Intrepid2::CellTools<PHX::Device>::getPhysicalSideNormals(normals_view, jacobian_side, side, *cellType);

      // scale normals (unity)
      Intrepid2::RealSpaceTools<PHX::Device>::vectorNorm(normal_lengths, normals_view, Intrepid2::NORM_TWO);
      Intrepid2::FunctionSpaceTools<PHX::Device>::scalarMultiplyDataData(normals_view, normal_lengths, normals_view, true);

      for (std::size_t iCell=0; iCell < numCells_; ++iCell)
        for (int icoor=0; icoor<numCellDims; ++icoor)
          for (int qp=0; qp<numSideQPs; ++qp)
            normals(sideSet.cellsOnSide(side,iCell),side,qp, icoor) = normals_view(iCell,qp,icoor);
    }
  }

}

} // Namespace PHAL
