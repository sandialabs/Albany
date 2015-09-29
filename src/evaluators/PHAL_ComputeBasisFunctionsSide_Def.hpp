//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace PHAL {

template<typename EvalT, typename Traits>
ComputeBasisFunctionsSide<EvalT, Traits>::
ComputeBasisFunctionsSide (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector ),
  inv_metric    (p.get<std::string> ("Inverse Metric Name"), dl->side_qp_tensor ),
  w_measure     (p.get<std::string> ("Weighted Measure Name"), dl->side_qp_scalar ),
  metric_det    (p.get<std::string> ("Metric Determinant Name"), dl->side_qp_scalar ),
  BF            (p.get<std::string> ("BF Name"), dl->side_node_qp_scalar),
  GradBF        (p.get<std::string> ("Gradient BF Name"), dl->side_node_qp_gradient)
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(w_measure);
  this->addEvaluatedField(metric_det);
  this->addEvaluatedField(inv_metric);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(GradBF);

  sideSetName = p.get<std::string>("Side Set Name");

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->side_node_qp_gradient->dimensions(dim);

  int numCells = dim[0];
  numSides     = dim[1];
  numSideNodes = dim[2];
  numSideQPs   = dim[3];
  sideDims     = dim[4];
  cellDims     = sideDims+1;


#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  *output << "Compute Basis Functions Side has: "
          << numCells << " cells, "
          << numSides << " sides, "
          << numSideNodes << " side nodes, "
          << numSideQPs << " side QPs, "
          << sideDims << " side dimensions.\n";
#endif

  // Allocate Temporary FieldContainers
  Intrepid::FieldContainer<RealType>  cub_points;
  cub_points.resize(numSideQPs,sideDims);
  cub_weights.resize(numSideQPs);
  val_at_cub_points.resize(numSideNodes, numSideQPs);
  grad_at_cub_points.resize(numSideNodes, numSideQPs, sideDims);
  cub_weights.resize(numSideQPs);
  tangents.resize(sideDims,cellDims,numSideQPs);
  metric.resize(numSideQPs,sideDims,sideDims);

  // Pre-Calculate reference element quantitites
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  cubature = p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature Side");
  cubature->getCubature(cub_points, cub_weights);

  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  intrepidBasis = p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis Side");
  intrepidBasis->getValues(val_at_cub_points, cub_points, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, cub_points, Intrepid::OPERATOR_GRAD);

  // Index of the nodes on the sides in the numeration of the cell
  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    sideNodes[side].resize(numSideNodes);
    for (int node=0; node<numSideNodes; ++node)
    {
      sideNodes[side][node] = cellType->getNodeMap(sideDims,side,node);
    }
  }

  this->setName("ComputeBasisFunctionsSide"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctionsSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(metric_det,fm);
  this->utils.setFieldData(inv_metric,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);

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
          for (int ider=0; ider<sideDims; ++ider)
          {
            GradBF(cell,side,node,qp,ider) = grad_at_cub_points(node,qp,ider);
          }
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
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

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
            tangents(itan,icoor,qp) += coordVec(cell,sideNodes[side][node],icoor) * grad_at_cub_points(node,qp,itan);
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
  }
}

} // Namespace PHAL
