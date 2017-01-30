//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
FieldNormBase<EvalT, Traits, ScalarT>::
FieldNormBase (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName = p.get<std::string> ("Field Name");
  std::string fieldNormName = p.get<std::string> ("Field Norm Name");

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

    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->cell_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_vector->dimensions(dims);
  }
  else if (layout=="Cell Side Gradient")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->cell_gradient);
    field_norm = decltype(field_norm)(fieldNormName, dl->cell_scalar2);

    dl->cell_gradient->dimensions(dims);
  }
  else if (layout=="Cell Side Node Vector")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->node_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->node_scalar);

    dl->node_vector->dimensions(dims);
  }
  else if (layout=="Cell Side QuadPoint Vector")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layouts structure does not appear to be that of a side set.\n");

    field      = decltype(field)(fieldName, dl->qp_vector);
    field_norm = decltype(field_norm)(fieldNormName, dl->qp_scalar);

    dl->qp_vector->dimensions(dims);
  }
  else if (layout=="Cell Side QuadPoint Gradient")
  {
    sideSetName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
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
  }
  else if (type=="Given Value")
  {
    regularization_type = GIVEN_VALUE;
    regularization = options.get<double>("Regularization Value");
  }
  else if (type=="Given Parameter")
  {
    regularization_type = GIVEN_PARAMETER;
    regularizationParam = decltype(regularizationParam)(options.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
  }
  else if (type=="Parameter Exponential")
  {
    regularization_type = PARAMETER_EXPONENTIAL;
    regularizationParam = decltype(regularizationParam)(options.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid regularization type");
  }

  numDims = dims.size();

  this->setName("FieldNormBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void FieldNormBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(field_norm,fm);

  if (regularization_type==GIVEN_PARAMETER || regularization_type==PARAMETER_EXPONENTIAL)
    this->utils.setFieldData(regularizationParam,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void FieldNormBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (regularization_type==GIVEN_PARAMETER)
    regularization = Albany::ScalarConverter<ScalarT>::apply(regularizationParam(0));
  else if (regularization_type==PARAMETER_EXPONENTIAL)
    regularization = pow(10.0, -10.0*Albany::ScalarConverter<ScalarT>::apply(regularizationParam(0)));

#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

    if (regularization_type!=NONE)
      if (std::fabs(printedReg-regularization)>1e-6*regularization)
      {
          *output << "[Field Norm<" << PHX::typeAsString<EvalT>() << ">]] reg = " << regularization << "\n";
          printedReg = regularization;
      }
#endif

  ScalarT norm;
  switch (numDims)
  {
    case 2:
      // Cell Vector/Gradient
      for (int cell(0); cell<dims[0]; ++cell)
      {
        norm = 0;
        for (int dim(0); dim<dims[1]; ++dim)
        {
          norm += std::pow(field(cell,dim),2);
        }
        field_norm(cell) = std::sqrt(norm + regularization);
      }
      break;
    case 3:
      // Cell Node/QuadPoint Vector/Gradient
      for (int cell(0); cell<dims[0]; ++cell)
      {
        norm = 0;
        for (int i(0); i<dims[1]; ++i)
        {
          for (int dim(0); dim<dims[2]; ++dim)
          {
            norm += std::pow(field(cell,i,dim),2);
          }
          field_norm(cell,i) = std::sqrt(norm + regularization);
        }
      }
      break;
    case 4:
      // Cell Side Node/QuadPoint Vector/Gradient
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

          norm = 0;
          for (int i(0); i<dims[2]; ++i)
          {
            for (int dim(0); dim<dims[3]; ++dim)
            {
              norm += std::pow(field(cell,side,i,dim),2);
            }
            field_norm(cell,side,i) = std::sqrt(norm + regularization);
          }
        }
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Invalid field layout.\n");
  }
}

} // Namespace FELIX
