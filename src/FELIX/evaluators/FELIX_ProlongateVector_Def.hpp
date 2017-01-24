//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
ProlongateVectorBase<EvalT, Traits, ScalarT>::
ProlongateVectorBase(const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl_in,
                     const Teuchos::RCP<Albany::Layouts>& dl_out)
{
  std::string name = p.get<std::string>("Field Name");

  std::string layout = p.get<std::string>("Field Layout");
  if (layout=="Cell Vector")
  {
    v_in  = PHX::MDField<ScalarT>(name,dl_in->cell_vector);
    v_out = PHX::MDField<ScalarT>(name,dl_out->cell_vector);
  }
  else if (layout=="Cell Node Vector")
  {
    v_in  = PHX::MDField<ScalarT>(name,dl_in->node_vector);
    v_out = PHX::MDField<ScalarT>(name,dl_out->node_vector);
  }
  else if (layout=="Cell QuadPoint Vector")
  {
    v_in  = PHX::MDField<ScalarT>(name,dl_in->qp_vector);
    v_out = PHX::MDField<ScalarT>(name,dl_out->qp_vector);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                "Error! Field layout not supported.\n");
  }

  this->addDependentField(v_in.fieldTag());
  this->addEvaluatedField(v_out);

  dims_in.resize(v_in.fieldTag().dataLayout().rank());
  dims_out.resize(v_out.fieldTag().dataLayout().rank());
  for (int i=0; i<dims_in.size()-1; ++i)
  {
    dims_in[i] = v_in.fieldTag().dataLayout().dimension(i);
    dims_out[i] = v_out.fieldTag().dataLayout().dimension(i);
    TEUCHOS_TEST_FOR_EXCEPTION (dims_in[i]!=dims_out[i], std::logic_error,
                              "Error! The number of Cell/Nodes/QuadPoints must be the same for both in/out vectors.\n");
  }
  TEUCHOS_TEST_FOR_EXCEPTION (dims_in.back()>dims_out.back(), std::logic_error,
                            "Error! The input vector dimension is larger than the output (this is not a prolongation).\n");

  pad_value = p.isParameter("Pad Value") ? p.get<double>("Pad Value") : 0.;
  pad_back = p.isParameter("Pad Back") ? p.get<bool>("Pad Back") : true;

  this->setName("ProlongateVectorBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void ProlongateVectorBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(v_in,fm);
  this->utils.setFieldData(v_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void ProlongateVectorBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  switch (dims_out.size())
  {
    case 2:
      if (pad_back)
      {
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int dim=0; dim<dims_in[1]; ++dim)
            v_out (cell,dim) = v_in(cell,dim);

          for (int dim=dims_in[1]; dim<dims_out[1]; ++dim)
            v_out (cell,dim) = pad_value;
        }
      }
      else
      {
        int offset = dims_out[1]-dims_in[1];
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int dim=0; dim<offset; ++dim)
            v_out (cell,dim) = pad_value;

          for (int dim=0; dim<dims_in[1]; ++dim)
            v_out (cell,dim+offset) = v_in(cell,dim);
        }
      }
      break;
    case 3:
      if (pad_back)
      {
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int pt=0; pt<dims_in[1]; ++pt)
          {
            for (int dim=0; dim<dims_in[2]; ++dim)
              v_out (cell,pt,dim) = v_in(cell,pt,dim);

            for (int dim=dims_in[2]; dim<dims_out[2]; ++dim)
              v_out (cell,pt,dim) = pad_value;
          }
        }
      }
      else
      {
        int offset = dims_out[1]-dims_in[1];
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int pt=0; pt<dims_in[1]; ++pt)
          {
            for (int dim=0; dim<offset; ++dim)
              v_out (cell,pt,dim) = pad_value;

            for (int dim=0; dim<dims_in[2]; ++dim)
              v_out (cell,pt,dim+offset) = v_in(cell,pt,dim);
          }
        }
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid field dimensions. An error should have already been raised.\n");
  }
}

} // Namespace FELIX
