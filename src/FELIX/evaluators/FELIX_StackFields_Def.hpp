//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
StackFieldsBase<EvalT, Traits, ScalarT>::
StackFieldsBase(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl_in_1,
                 const Teuchos::RCP<Albany::Layouts>& dl_in_2,
                 const Teuchos::RCP<Albany::Layouts>& dl_out)
{
  std::string str_layout_out = p.get<std::string>("Stacked Field Layout");
  Teuchos::RCP<PHX::DataLayout> layout_out = getLayout(str_layout_out, dl_out);
  rank_out = layout_out->rank();
  std::string last_tag = layout_out->name(rank_out-1);
  TEUCHOS_TEST_FOR_EXCEPTION (last_tag!="Vector" && last_tag!="Gradient", std::logic_error,
                              "Error! Output stacked field must be xxx_vector or xxx_gradient.\n");
  dims_out.resize(rank_out);
  for (int i=0; i<rank_out; ++i)
    dims_out[i] = layout_out->dimension(i);

  field_out = PHX::MDField<ScalarT>(p.get<std::string>("Stacked Field Name"),layout_out);
  this->addEvaluatedField(field_out);

  num_fields_in = 2;
  fields_in.reserve(2);
  ranks_in.resize(2);
  dims_in.resize(2);
  offsets.resize(3);

  std::string name_1 = p.get<std::string>("First Field Name");
  std::string name_2 = p.get<std::string>("Second Field Name");

  std::string str_layout_1 = p.get<std::string>("First Field Layout");
  std::string str_layout_2 = p.get<std::string>("Second Field Layout");

  Teuchos::RCP<PHX::DataLayout> layout_1 = getLayout(str_layout_1, dl_in_1);
  Teuchos::RCP<PHX::DataLayout> layout_2 = getLayout(str_layout_2, dl_in_2);

  TEUCHOS_TEST_FOR_EXCEPTION (!isCompatible(layout_1,layout_out), std::logic_error,
                              "Error! The number of Cell/Node/QuadPoint must be the same across fields.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (!isCompatible(layout_2,layout_out), std::logic_error,
                              "Error! The number of Cell/Node/QuadPoint must be the same across fields.\n");

  ranks_in[0] = layout_1->rank();
  ranks_in[1] = layout_2->rank();

  dims_in[0] = ranks_in[0]==rank_out ? layout_1->dimension(layout_1->rank()-1) : 1;
  dims_in[1] = ranks_in[1]==rank_out ? layout_2->dimension(layout_2->rank()-1) : 1;

  offsets[0] = 0;
  offsets[1] = dims_in[0];
  offsets[2] = dims_in[0]+dims_in[1];

  fields_in.push_back(PHX::MDField<ScalarT>(name_1,layout_1));
  fields_in.push_back(PHX::MDField<ScalarT>(name_2,layout_2));

  this->addDependentField(fields_in[0]);
  this->addDependentField(fields_in[1]);

  TEUCHOS_TEST_FOR_EXCEPTION (offsets[2]!=dims_out.back(), std::logic_error,
                              "Error! The sum of input fields dimensions does not match the output field dimension.\n");

  this->setName("StackFieldsBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
StackFieldsBase<EvalT, Traits, ScalarT>::
StackFieldsBase(const Teuchos::ParameterList& p,
                 const std::vector<Teuchos::RCP<Albany::Layouts>>& dls_in,
                 const Teuchos::RCP<Albany::Layouts>& dl_out)
{
  std::string str_layout_out = p.get<std::string>("Stacked Field Layout");
  Teuchos::RCP<PHX::DataLayout> layout_out = getLayout(str_layout_out, dl_out);
  rank_out = layout_out->rank();
  std::string last_tag = layout_out->name(rank_out-1);
  TEUCHOS_TEST_FOR_EXCEPTION (last_tag!="Vector" && last_tag!="Gradient", std::logic_error,
                              "Error! Output stacked field must be xxx_vector or xxx_gradient.\n");
  dims_out.resize(rank_out);
  for (int i=0; i<rank_out; ++i)
    dims_out[i] = layout_out->dimension(i);

  field_out = PHX::MDField<ScalarT>(p.get<std::string>("Stacked Field Name"),layout_out);
  this->addEvaluatedField(field_out);

  num_fields_in = dls_in.size();
  fields_in.resize(num_fields_in);
  dims_in.resize(num_fields_in);
  ranks_in.resize(num_fields_in);
  offsets.resize(num_fields_in+1);

  Teuchos::Array<std::string> names = p.get<Teuchos::Array<std::string>>("Fields Names");
  Teuchos::Array<std::string> str_layouts = p.get<Teuchos::Array<std::string>>("Fields Layouts");

  TEUCHOS_TEST_FOR_EXCEPTION (names.size()!=num_fields_in, Teuchos::Exceptions::InvalidParameter,
                              "Error! Input names array has the wrong size.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (str_layouts.size()!=num_fields_in, Teuchos::Exceptions::InvalidParameter,
                              "Error! Input layouts array has the wrong size.\n");

  offsets[0] = 0;
  for (int i=0; i<num_fields_in; ++i)
  {
    Teuchos::RCP<PHX::DataLayout> layout_i = getLayout(str_layouts[i],dls_in[i]);
    TEUCHOS_TEST_FOR_EXCEPTION (!isCompatible(layout_i,layout_out), std::logic_error,
                                "Error! The number of Cell/Node/QuadPoint must be the same across fields.\n");

    ranks_in[i] = layout_i->rank();
    dims_in[i] = ranks_in[i]==rank_out ? layout_i->dimension(ranks_in[i]-1) : 1;
    offsets[i+1] = offsets[i] + dims_in[i];

    fields_in[i] = PHX::MDField<ScalarT>(names[i],layout_i);
    this->addDependentField(fields_in[i]);
  }

  TEUCHOS_TEST_FOR_EXCEPTION (offsets.back()!=dims_out.back(), std::logic_error,
                              "Error! The sum of input fields dimensions does not match the output field dimension.\n");

  this->setName("StackFieldsBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void StackFieldsBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (int i=0; i<num_fields_in; ++i)
    this->utils.setFieldData(fields_in[i],fm);

  this->utils.setFieldData(field_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void StackFieldsBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  switch (rank_out)
  {
    case 2:
      for (int i=0; i<num_fields_in; ++i)
        if (ranks_in[i]==rank_out)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int dim=0; dim<dims_in[i]; ++dim)
              field_out(cell,dim+offsets[i]) = fields_in[i](cell,dim);
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            field_out(cell,offsets[i]) = fields_in[i](cell);
      break;
    case 3:
      for (int i=0; i<num_fields_in; ++i)
        if (ranks_in[i]==rank_out)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int j=0; j<dims_out[1]; ++j)
              for (int dim=0; dim<dims_in[i]; ++dim)
                field_out(cell,j,dim+offsets[i]) = fields_in[i](cell,j,dim);
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int j=0; j<dims_out[1]; ++j)
              field_out(cell,j,offsets[i]) = fields_in[i](cell,j);
      break;
    case 4:
      for (int i=0; i<num_fields_in; ++i)
        if (ranks_in[i]==rank_out)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int j=0; j<dims_out[1]; ++j)
              for (int k=0; k<dims_out[2]; ++k)
                for (int dim=0; dim<dims_in[i]; ++dim)
                  field_out(cell,j,k,dim+offsets[i]) = fields_in[i](cell,j,k,dim);
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int j=0; j<dims_out[1]; ++j)
              for (int k=0; k<dims_out[2]; ++k)
                field_out(cell,j,k,offsets[i]) = fields_in[i](cell,j,k);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid output field rank. An error should already have been raised.\n");
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
Teuchos::RCP<PHX::DataLayout>
StackFieldsBase<EvalT, Traits, ScalarT>::
getLayout (const std::string& name,
           const Teuchos::RCP<Albany::Layouts>& dl) const
{
  Teuchos::RCP<PHX::DataLayout> layout;
  if (name=="Cell Scalar")
    layout = dl->cell_scalar2;
  else if (name=="Cell Vector")
    layout = dl->cell_vector;
  else if (name=="Cell Gradient")
    layout = dl->cell_gradient;
  else if (name=="Cell Node Scalar")
    layout = dl->node_scalar;
  else if (name=="Cell Node Vector")
    layout = dl->node_vector;
  else if (name=="Cell Node Gradient")
    layout = dl->node_gradient;
  else if (name=="Cell QuadPoint Scalar")
    layout = dl->qp_scalar;
  else if (name=="Cell QuadPoint Vector")
    layout = dl->qp_vector;
  else if (name=="Cell QuadPoint Gradient")
    layout = dl->qp_gradient;
  else
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Layout not supported.\n");

  return layout;
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
bool StackFieldsBase<EvalT, Traits, ScalarT>::
isCompatible (const Teuchos::RCP<PHX::DataLayout>& layout_1,
              const Teuchos::RCP<PHX::DataLayout>& layout_2) const
{
  int rank_1 = layout_1->rank();
  int rank_2 = layout_2->rank();

  if (rank_1==rank_2)
  {
    for (int i=0; i<rank_1-1; ++i)
      if (layout_1->name(i)!=layout_2->name(i))
        return false;
  }
  else if (rank_1==rank_2-1)
  {
    for (int i=0; i<rank_1; ++i)
      if (layout_1->name(i)!=layout_2->name(i))
        return false;
  }
  else if (rank_2==rank_1-1)
  {
    for (int i=0; i<rank_2; ++i)
      if (layout_1->name(i)!=layout_2->name(i))
        return false;
  }
  else
  {
    return false;
  }

  return true;
}

} // Namespace FELIX
