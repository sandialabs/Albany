//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "PHAL_FieldFrobeniusNorm.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_SacadoTypes.hpp"

//uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::
FieldFrobeniusNormBase (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName = p.get<std::string> ("Field Name");
  std::string fieldNormName = p.get<std::string> ("Field Norm Name");

  eval_on_side = dl->isSideLayouts;

  std::string layout = p.get<std::string>("Field Layout");
  if (layout=="Cell Vector")
  {
    field      = decltype(field)(fieldName, dl->cell_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_vector->dimensions(dims);
  }
  else if (layout=="Cell Gradient")
  {
    field      = decltype(field)(fieldName, dl->cell_gradient);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_gradient->dimensions(dims);
  }
  else if (layout=="Cell Node Vector")
  {
    field      = decltype(field)(fieldName, dl->node_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->node_scalar);

    dl->node_vector->dimensions(dims);
  }
  else if (layout=="Cell QuadPoint Vector")
  {
    field      = decltype(field)(fieldName, dl->qp_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->qp_scalar);

    dl->qp_vector->dimensions(dims);
  }
  else if (layout=="Cell QuadPoint Gradient")
  {
    field      = decltype(field)(fieldName, dl->qp_gradient);
    field_norm = decltype(field_norm)(fieldNormName, dl->qp_scalar);

    dl->qp_gradient->dimensions(dims);
  }
  else if (layout=="Cell Side Vector")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!eval_on_side, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->cell_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_vector->dimensions(dims);
  }
  else if (layout=="Cell Side Gradient")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!eval_on_side, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->cell_gradient);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_gradient->dimensions(dims);
  }
  else if (layout=="Cell Side Node Vector")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!eval_on_side, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->node_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->node_scalar);

    dl->node_vector->dimensions(dims);
  }
  else if (layout=="Cell Side QuadPoint Vector")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!eval_on_side, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->qp_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->qp_scalar);

    dl->qp_vector->dimensions(dims);
  }
  else if (layout=="Cell Side QuadPoint Gradient")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!eval_on_side, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->qp_gradient);
    field_norm = decltype(field_norm)(fieldNormName, dl->qp_scalar);

    dl->qp_gradient->dimensions(dims);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid field layout.\n");
  }

  this->addDependentField(field);
  this->addEvaluatedField(field_norm);

  Teuchos::ParameterList& options = p.get<Teuchos::ParameterList*>("Parameter List")->sublist(fieldNormName);
  std::string type = options.get<std::string>("Regularization Type","None");
  if (type=="None")
  {
    regularization_type = NONE;
    regularization = 0.0;
  }
  else if (type=="Given Value")
  {
    regularization_type = GIVEN_VALUE;
    regularization = options.get<double>("Regularization Value");
    printedReg = -1.0;
  }
  else if (type=="Given Parameter")
  {
    regularization_type = GIVEN_PARAMETER;
    regularizationParam = decltype(regularizationParam)(options.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
    printedReg = -1.0;
  }
  else if (type=="Parameter Exponential")
  {
    regularization_type = PARAMETER_EXPONENTIAL;
    regularizationParam = decltype(regularizationParam)(options.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
    printedReg = -1.0;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid regularization type");
  }

  numDims = dims.size();

  TEUCHOS_TEST_FOR_EXCEPTION (numDims > 4, Teuchos::Exceptions::InvalidParameter, "Error! Layout has more dimensions than expected");

  for (int i = 0; i < numDims; ++i)
    dimsArray[i] = dims[i];

  this->setName("FieldFrobeniusNormBase(" + fieldNormName + ")" + PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(field_norm,fm);

  if (regularization_type==GIVEN_PARAMETER || regularization_type==PARAMETER_EXPONENTIAL)
    this->utils.setFieldData(regularizationParam,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::
operator() (const Dim2_Tag&, const int& sideSet_idx) const {

  ScalarT norm = 0;
  for (unsigned int dim(0); dim<dimsArray[1]; ++dim)
  {
    norm += std::pow(field(sideSet_idx,dim),2);
  }
  field_norm(sideSet_idx) = std::sqrt(norm + regularization);

}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::
operator() (const Dim3_Tag&, const int& sideSet_idx) const {

  ScalarT norm;
  for (unsigned int i(0); i<dimsArray[1]; ++i)
  {
    norm = 0;
    for (unsigned int dim(0); dim<dimsArray[2]; ++dim)
    {
      norm += std::pow(field(sideSet_idx,i,dim),2);
    }
    field_norm(sideSet_idx,i) = std::sqrt(norm + regularization);
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (regularization_type==GIVEN_PARAMETER)
    regularization = Albany::convertScalar<const ScalarT>(regularizationParam(0));
  else if (regularization_type==PARAMETER_EXPONENTIAL)
    regularization = pow(10.0, -10.0*Albany::convertScalar<const ScalarT>(regularizationParam(0)));

#ifdef OUTPUT_TO_SCREEN
  if (regularization_type!=NONE)
  {
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    if (std::fabs(printedReg-regularization)>1e-6*regularization)
    {
        *output << "[Field Norm<" << PHX::print<EvalT>() << ">]] reg = " << regularization << "\n";
        printedReg = regularization;
    }
  }
#endif

  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;

  sideSet = workset.sideSetViews->at(sideSetName);

  switch (numDims)
  {
    case 2:
      // <sideSet_idx,Vector/Gradient>
      Kokkos::parallel_for(Dim2_Policy(0,sideSet.size),*this);
      break;
    case 3:
      // <sideSet_idx,Node/QuadPoint,Vector/Gradient>
      Kokkos::parallel_for(Dim3_Policy(0,sideSet.size),*this);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Invalid field layout.\n");
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void FieldFrobeniusNormBase<EvalT, Traits, ScalarT>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT norm;
  switch (numDims)
  {
    case 2:
      // <Cell,Vector/Gradient>
      for (unsigned int cell(0); cell<workset.numCells; ++cell)
      {
        norm = 0;
        for (unsigned int dim(0); dim<dims[1]; ++dim)
        {
          norm += std::pow(field(cell,dim),2);
        }
        field_norm(cell) = std::sqrt(norm + regularization);
      }
      break;
    case 3:
      // <Cell,Node/QuadPoint,Vector/Gradient>
      for (unsigned int cell(0); cell<workset.numCells; ++cell)
      {
        for (unsigned int i(0); i<dims[1]; ++i)
        {
          norm = 0;
          for (unsigned int dim(0); dim<dims[2]; ++dim)
          {
            norm += std::pow(field(cell,i,dim),2);
          }
          field_norm(cell,i) = std::sqrt(norm + regularization);
        }
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Invalid field layout.\n");
  }
}

} // namespace PHAL
