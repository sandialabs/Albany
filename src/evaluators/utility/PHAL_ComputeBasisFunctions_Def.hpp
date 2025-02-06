//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec          (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature          (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  intrepid2FEBasis  (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 FE Basis") ),
  intrepid2MapBasis (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Ref-To-Phys Map Basis") ),
  weighted_measure  (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  BF                (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF               (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF            (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF           (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{
  this->addDependentField(coordVec.fieldTag());
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numCells = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];


  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  this->setName("ComputeBasisFunctions"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  jacobian_det = Kokkos::createDynRankView(coordVec.get_view(), "jacobian_det", numCells, numQPs);
  jacobian = Kokkos::createDynRankView(coordVec.get_view(), "jacobian", numCells, numQPs, numDims, numDims);
  jacobian_inv = Kokkos::createDynRankView(coordVec.get_view(), "jacobian_inv", numCells, numQPs, numDims, numDims);

  // Allocate Temporary Kokkos Views
  val_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("val_at_cub_points", numNodes, numQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("grad_at_cub_points", numNodes, numQPs, numDims);
  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("refPoints", numQPs, numDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("refWeights", numQPs);

  // Pre-Calculate reference element quantities
  cubature->getCubature(refPoints, refWeights);
  intrepid2FEBasis->getValues(val_at_cub_points, refPoints, Intrepid2::OPERATOR_VALUE);
  intrepid2FEBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  /** The allocated size of the Field Containers must currently
    * match the full workset size of the allocated PHX Fields,
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */

  typedef typename Intrepid2::CellTools<PHX::Device>   ICT;
  typedef Intrepid2::FunctionSpaceTools<PHX::Device>   IFST;

  ICT::setJacobian(jacobian, refPoints, coordVec.get_view(), intrepid2MapBasis);
  ICT::setJacobianInv (jacobian_inv, jacobian);
  ICT::setJacobianDet (jacobian_det, jacobian);

  bool isJacobianDetNegative =
    IFST::computeCellMeasure (weighted_measure.get_view(), jacobian_det, refWeights);
  IFST::HGRADtransformVALUE(BF.get_view(), val_at_cub_points);
  IFST::multiplyMeasure    (wBF.get_view(), weighted_measure.get_view(), BF.get_view());
  IFST::HGRADtransformGRAD (GradBF.get_view(), jacobian_inv, grad_at_cub_points);
  IFST::multiplyMeasure    (wGradBF.get_view(), weighted_measure.get_view(), GradBF.get_view());

  (void)isJacobianDetNegative;

  //debug {
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    std::cout << "cell " << cell << "\n";
    for (size_t v=0; v<numNodes; ++v) {
      std::cout << "  node " << v << ":";
      for (size_t dim=0; dim<numDims; ++dim) {
        std::cout << " " << coordVec(cell,v,dim);
      }
      std::cout << "\n";
    }
  }
  // }
}

//**********************************************************************
} // namespace PHAL
