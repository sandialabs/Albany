//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
NodesToCellInterpolationBase (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string interpType = p.get<std::string>("Interpolation Type");
  TEUCHOS_TEST_FOR_EXCEPTION ((interpType != "Value At Cell Barycenter") && (interpType != "Cell Average"),
      std::runtime_error, "Interpolation Type can either be \"Value At Cell Barycenter\" or \"Cell Average\"");

  interpolationType = (interpType == "Value At Cell Barycenter") ?
      ValueAtCellBarycenter : CellAverage;

  if(interpolationType == ValueAtCellBarycenter) {
    intrepidBasis = p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis");
    TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::is_null(intrepidBasis),
          std::runtime_error, "intrepidBasis needs to be nonnull when Interpolation Type is \"Value At Cell Barycenter\"");
  } else {
    BF = decltype(BF)(p.get<std::string>("BF Variable Name"), dl->node_qp_scalar);
    w_measure = decltype(w_measure)(p.get<std::string>("Weighted Measure Name"), dl->qp_scalar);
  }

  isVectorField = p.get<bool>("Is Vector Field");
  if (isVectorField)
  {
    field_node = decltype(field_node)(p.get<std::string> ("Field Node Name"), dl->node_vector);
    field_cell = decltype(field_cell)(p.get<std::string> ("Field Cell Name"), dl->cell_vector);

    vecDim = dl->node_vector->extent(2);
  }
  else
  {
    field_node = decltype(field_node)(p.get<std::string> ("Field Node Name"), dl->node_scalar);
    field_cell = decltype(field_cell)(p.get<std::string> ("Field Cell Name"), dl->cell_scalar2);
  }

  numQPs   = dl->qp_scalar->extent(1);
  numNodes = dl->node_scalar->extent(1);

  if(interpolationType == CellAverage) {
    this->addDependentField (BF.fieldTag());
    this->addDependentField (w_measure.fieldTag());
  }
  this->addDependentField (field_node.fieldTag());

  this->addEvaluatedField (field_cell);

  this->setName("NodesToCellInterpolation"+PHX::print<EvalT>());
}

//**********************************************************************
// Kokkos operators
template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Average_Vector_Field_Tag& tag, const int& cell) const{

  ScalarT field_qp_device;

  MeshScalarT meas_device = 0.0;
  for (int qp(0); qp<numQPs; ++qp)
  {
    meas_device += w_measure(cell,qp);
  }

  for (int dim(0); dim<vecDim; ++dim)
  {
    field_cell(cell,dim) = 0;
    for (int qp(0); qp<numQPs; ++qp)
    {
      field_qp_device = 0;
      for (int node(0); node<numNodes; ++node)
        field_qp_device += field_node(cell,node,dim)*BF(cell,node,qp);
      field_cell(cell,dim) += field_qp_device*w_measure(cell,qp);
    }
    field_cell(cell,dim) /= meas_device;
  }

}

template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Average_Scalar_Field_Tag& tag, const int& cell) const{

  ScalarT field_qp_device;

  MeshScalarT meas_device = 0.0;
  for (int qp(0); qp<numQPs; ++qp)
  {
    meas_device += w_measure(cell,qp);
  }

  field_cell(cell) = 0;
  for (int qp(0); qp<numQPs; ++qp)
  {
    field_qp_device = 0;
    for (int node(0); node<numNodes; ++node)
      field_qp_device += field_node(cell,node)*BF(cell,node,qp);
    field_cell(cell) += field_qp_device*w_measure(cell,qp);
  }
  field_cell(cell) /= meas_device;

}

template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Barycenter_Vector_Field_Tag& tag, const int& cell) const {
  
  for (int dim = 0; dim<vecDim; ++dim)
  {
    field_cell(cell,dim) = 0;
    for (int node = 0; node<numNodes; ++node)
      field_cell(cell,dim) += field_node(cell,node,dim)*basis_at_barycenter(node,0);
  }

}

template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Cell_Barycenter_Scalar_Field_Tag& tag, const int& cell) const {
  
  field_cell(cell) = 0;
  for (int node = 0; node<numNodes; ++node)
    field_cell(cell) += field_node(cell,node)*basis_at_barycenter(node,0);

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if(interpolationType == CellAverage) {
    this->utils.setFieldData(BF,fm);
    this->utils.setFieldData(w_measure,fm);
  }
  this->utils.setFieldData(field_node,fm);

  this->utils.setFieldData(field_cell,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();

  if(interpolationType == ValueAtCellBarycenter) {
    basis_at_barycenter = Kokkos::DynRankView<RealType, PHX::Device>("basis_at_barycenter", numNodes, 1);
    auto cellTopology = intrepidBasis->getBaseCellTopology();
    Kokkos::DynRankView<RealType, PHX::Device> refPoint("refPoint", 1, cellTopology.getDimension());
    Kokkos::DynRankView<RealType, PHX::Device> refWeight("refWeights", 1);

    // Pre-Calculate reference element quantities
    Intrepid2::DefaultCubatureFactory cubFactory;
    auto onePointCubature = cubFactory.create<PHX::Device, RealType, RealType>(cellTopology, 0);
    onePointCubature->getCubature(refPoint, refWeight);
    intrepidBasis->getValues(basis_at_barycenter, refPoint, Intrepid2::OPERATOR_VALUE);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void NodesToCellInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (interpolationType == CellAverage) {
    if (isVectorField) {
      Kokkos::parallel_for(Cell_Average_Vector_Field_Policy(0, workset.numCells), *this);
    } else {
      Kokkos::parallel_for(Cell_Average_Scalar_Field_Policy(0, workset.numCells), *this);
    }
  } else {
    if (isVectorField) {
      Kokkos::parallel_for(Cell_Barycenter_Vector_Field_Policy(0, workset.numCells), *this);
    } else {
      Kokkos::parallel_for(Cell_Barycenter_Scalar_Field_Policy(0, workset.numCells), *this);
    }
  }
#else
  MeshScalarT meas;
  ScalarT field_qp;

  for (int cell=0; cell<workset.numCells; ++cell)
  {
    if(interpolationType == CellAverage) {
      meas = 0.0;
      for (int qp(0); qp<numQPs; ++qp)
      {
        meas += w_measure(cell,qp);
      }

      if (isVectorField)
      {
        for (int dim(0); dim<vecDim; ++dim)
        {
          field_cell(cell,dim) = 0;
          for (int qp(0); qp<numQPs; ++qp)
          {
            field_qp = 0;
            for (int node(0); node<numNodes; ++node)
              field_qp += field_node(cell,node,dim)*BF(cell,node,qp);
            field_cell(cell,dim) += field_qp*w_measure(cell,qp);
          }
          field_cell(cell,dim) /= meas;
        }
      }
      else // scalarField
      {
        field_cell(cell) = 0;
        for (int qp(0); qp<numQPs; ++qp)
        {
          field_qp = 0;
          for (int node(0); node<numNodes; ++node)
            field_qp += field_node(cell,node)*BF(cell,node,qp);
          field_cell(cell) += field_qp*w_measure(cell,qp);
        }
        field_cell(cell) /= meas;
      }
    } else  //if(interpolationType == ValueAtCellBarycenter)
    {
      if (isVectorField)
      {
        for (int dim(0); dim<vecDim; ++dim)
        {
          field_cell(cell,dim) = 0;
          for (int node(0); node<numNodes; ++node)
            field_cell(cell,dim) += field_node(cell,node,dim)*basis_at_barycenter(node,0);
        }
      }
      else // scalarField
      {
        field_cell(cell) = 0;
        for (int node(0); node<numNodes; ++node)
          field_cell(cell) += field_node(cell,node)*basis_at_barycenter(node,0);
      }
    }
  }
#endif
}

} // Namespace PHAL
