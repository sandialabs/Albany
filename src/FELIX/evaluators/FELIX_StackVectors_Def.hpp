//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
StackVectorsBase<EvalT, Traits, ScalarT>::
StackVectorsBase(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl_in_1,
                 const Teuchos::RCP<Albany::Layouts>& dl_in_2,
                 const Teuchos::RCP<Albany::Layouts>& dl_out)
{
  v_out = PHX::MDField<ScalarT>(p.get<std::string>("Stacked Vector Name"),dl_out->node_vector);
  this->addEvaluatedField(v_out);

  numNodes = dl_out->node_vector->dimension(1);
  TEUCHOS_TEST_FOR_EXCEPTION (dl_in_1->node_scalar->dimension(1)!=numNodes, std::logic_error,
                              "Error! The number of nodes must be the same across residuals.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (dl_in_2->node_scalar->dimension(1)!=numNodes, std::logic_error,
                              "Error! The number of nodes must be the same across residuals.\n");

  num_v_in = 2;
  v_in.reserve(2);
  ranks_in.resize(2);
  dims_in.resize(2);
  offsets.resize(3);

  std::string name_1 = p.get<std::string>("First Vector Name");
  std::string name_2 = p.get<std::string>("Second Vector Name");
  ranks_in[0] = p.get<int>("First Residual Rank");
  ranks_in[1] = p.get<int>("Second Residual Rank");

  offsets[0] = 0;
  if (ranks_in[0]==0)
  {
    v_in.push_back(PHX::MDField<ScalarT>(name_1,dl_in_1->node_scalar));
    dims_in[0] = 1;
  }
  else if (ranks_in[0]==1)
  {
    v_in.push_back(PHX::MDField<ScalarT>(name_1,dl_in_1->node_vector));
    dims_in[0] = dl_in_1->node_vector->dimension(2);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Only scalar/vector residuals can be stacked.\n");
  }

  offsets[1] = dims_in[0];
  if (ranks_in[1]==0)
  {
    v_in.push_back(PHX::MDField<ScalarT>(name_2,dl_in_2->node_scalar));
    dims_in[1] = 1;
  }
  else if (ranks_in[1]==1)
  {
    v_in.push_back(PHX::MDField<ScalarT>(name_2,dl_in_2->node_vector));
    dims_in[1] = dl_in_2->node_vector->dimension(2);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Only scalar/vector residuals can be stacked.\n");
  }

  this->addDependentField(v_in[0]);
  this->addDependentField(v_in[1]);

  TEUCHOS_TEST_FOR_EXCEPTION (dims_in[0]+dims_in[1]!=dl_out->node_vector->dimension(2), std::logic_error,
                              "Error! The sum of input residuals dimension does not match the output residual dimension.\n");

  this->setName("StackVectorsBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
StackVectorsBase<EvalT, Traits, ScalarT>::
StackVectorsBase(const Teuchos::ParameterList& p,
                 const std::vector<Teuchos::RCP<Albany::Layouts>>& dls_in,
                 const Teuchos::RCP<Albany::Layouts>& dl_out)
{
  v_out = PHX::MDField<ScalarT>(p.get<std::string>("Stacked Vector Name"),dl_out->node_vector);
  this->addEvaluatedField(v_out);

  numNodes = dl_out->node_vector->dimension(1);

  num_v_in = dls_in.size();
  v_in.reserve(num_v_in);
  dims_in.resize(num_v_in);
  offsets.resize(num_v_in+1);

  Teuchos::Array<std::string> names = p.get<Teuchos::Array<std::string>>("VectorsBase Names");
  ranks_in = p.get<std::vector<int>>("VectorsBase Ranks");
  TEUCHOS_TEST_FOR_EXCEPTION (ranks_in.size()!=num_v_in, Teuchos::Exceptions::InvalidParameter,
                              "Error! Input ranks vector has the wrong size.\n");

  int check_sum = 0;
  offsets[0] = 0;
  for (int k=0; k<num_v_in; ++k)
  {
    PHX::MDField<ScalarT> res_k;
    if (ranks_in[k]==0)
    {
      res_k = PHX::MDField<ScalarT>(names[k],dls_in[k]->node_scalar);
      dims_in[k] = 1;
    }
    else if (ranks_in[k]==1)
    {
      res_k = PHX::MDField<ScalarT>(names[k],dls_in[k]->node_vector);
      dims_in[k] = dls_in[k]->node_vector->dimension(2);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Only scalar/vector residuals can be stacked.\n");
    }

    TEUCHOS_TEST_FOR_EXCEPTION (dls_in[k]->node_scalar->dimension(1)!=numNodes, std::logic_error,
                                "Error! The number of nodes must be the same across residuals.\n");

    offsets[k+1] = offsets[k]+dims_in[k];
    check_sum += dims_in[k];
    v_in.push_back(res_k);
    this->addDependentField(v_in[k]);
  }

  TEUCHOS_TEST_FOR_EXCEPTION (check_sum!=dl_out->node_vector->dimension(2), std::logic_error,
                              "Error! The sum of input residuals dimension does not match the output residual dimension.\n");

  this->setName("StackVectorsBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void StackVectorsBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (int k=0; k<num_v_in; ++k)
    this->utils.setFieldData(v_in[k],fm);

  this->utils.setFieldData(v_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void StackVectorsBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int k=0; k<num_v_in; ++k)
  {
    switch (ranks_in[k])
    {
      case 0:
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int node=0; node<numNodes; ++node)
          {
            v_out (cell,node,offsets[k]) = v_in[k](cell,node);
          }
        }
        break;
      case 1:
        for (int cell=0; cell<workset.numCells; ++cell)
        {
          for (int node=0; node<numNodes; ++node)
          {
            for (int dim=0; dim<dims_in[k]; ++dim)
              v_out (cell,node,offsets[k]+dim) = v_in[k](cell,node,dim);
          }
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid residual rank. However, an error should have already been raised.\n");
    }
  }
}

} // Namespace FELIX
