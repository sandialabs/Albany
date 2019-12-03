//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_SimpleOperationEvaluator.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename OutScalarT, typename Operation>
SimpleOperationBase<EvalT, Traits, InScalarT, OutScalarT, Operation>::
SimpleOperationBase (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldInName  = p.get<std::string> ("Input Field Name");
  std::string fieldOutName = p.get<std::string> ("Output Field Name");

  auto layout = Albany::getLayout(p.get<std::string>("Field Layout"),*dl);
  TEUCHOS_TEST_FOR_EXCEPTION (layout.is_null(), std::runtime_error, "Error! Input layout is null.\n");

  field_in  = PHX::MDField<const InScalarT> (fieldInName,  layout);
  field_out = PHX::MDField<OutScalarT> (fieldOutName, layout);

  this->addDependentField(field_in);
  this->addEvaluatedField(field_out);

  this->setName("SimpleOperation"+PHX::typeAsString<Operation>()+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> class UnaryOperation>
SimpleUnaryOperation<EvalT, Traits, InOutScalarT, UnaryOperation>::
SimpleUnaryOperation (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InOutScalarT,InOutScalarT,UnaryOperation<InOutScalarT>> (p,dl)
{
  this->op.setup(p);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> class UnaryOperation>
void SimpleUnaryOperation<EvalT, Traits, InOutScalarT, UnaryOperation>::
evaluateFields (typename Traits::EvalData /* workset */)
{
  PHAL::MDFieldIterator<const InOutScalarT> in(this->field_in);
  PHAL::MDFieldIterator<InOutScalarT> out(this->field_out);
  for (; !in.done(); ++in, ++out) {
    *out = this->op(*in);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryScaleOp<EvalT, Traits, InOutScalarT>::
UnaryScaleOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Scale>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryLogOp<EvalT, Traits, InOutScalarT>::
UnaryLogOp (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Log>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryExpOp<EvalT, Traits, InOutScalarT>::
UnaryExpOp (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Exp>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryLowPassOp<EvalT, Traits, InOutScalarT>::
UnaryLowPassOp (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::LowPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryHighPassOp<EvalT, Traits, InOutScalarT>::
UnaryHighPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::HighPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryBandPassOp<EvalT, Traits, InOutScalarT>::
UnaryBandPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::BandPass>(p,dl) {}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT, template<typename,typename> class BinaryOperation>
SimpleBinaryOperation<EvalT, Traits, InScalarT, FieldScalarT, BinaryOperation>::
SimpleBinaryOperation (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InScalarT,
                      typename Albany::StrongestScalarType<InScalarT,
                                                           FieldScalarT>::type,
                      BinaryOperation<InScalarT,FieldScalarT>> (p,dl)

{
  auto layout_inout = Albany::getLayout(p.get<std::string>("Field Layout"),*dl);
  auto layout_field = Albany::getLayout(p.get<std::string>("Parameter Field Layout"),*dl);
  param_field = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field Name"),layout_field);

  this->addDependentField(param_field);

  auto inout_rank = layout_inout->rank();
  auto field_rank = layout_field->rank();
  if (inout_rank==field_rank) {
    TEUCHOS_TEST_FOR_EXCEPTION(*layout_inout!=*layout_field, std::logic_error,
                               "Error! Incompatible layouts (" + layout_inout->identifier() + " and " + layout_field->identifier() + ".\n");
    sizes_ratio = 1;
  } else if (inout_rank==(field_rank+1)) {
    for (int i=0; i<field_rank; ++i) {
      TEUCHOS_TEST_FOR_EXCEPTION(layout_inout->name(i)!=layout_field->name(i), std::logic_error,
                                 "Error! Incompatible layouts (" + layout_inout->identifier() + " and " + layout_field->identifier() + ".\n");
    }
    sizes_ratio = layout_inout->extent_int(layout_inout->rank()-1);
  } else if (inout_rank==(field_rank+2)) {
    for (int i=0; i<field_rank; ++i) {
      TEUCHOS_TEST_FOR_EXCEPTION(layout_inout->name(i)!=layout_field->name(i), std::logic_error,
                                 "Error! Incompatible layouts (" + layout_inout->identifier() + " and " + layout_field->identifier() + ".\n");
    }
    sizes_ratio = layout_inout->extent_int(layout_inout->rank()-1) *
                  layout_inout->extent_int(layout_inout->rank()-2);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT, template<typename,typename> class BinaryOperation>
void SimpleBinaryOperation<EvalT, Traits, InScalarT, FieldScalarT, BinaryOperation>::
evaluateFields (typename Traits::EvalData /* workset */)
{
  PHAL::MDFieldIterator<const InScalarT> in(this->field_in);
  PHAL::MDFieldIterator<const FieldScalarT> param(param_field);
  PHAL::MDFieldIterator<OutScalarT> out(this->field_out);

  if (sizes_ratio==1) {
    for (; !in.done(); ++in, ++out, ++param) {
      *out = this->op(*in,*param);
    }
  } else {
    for (; !in.done(); ++param) {
      for (int i=0; i<sizes_ratio; ++in, ++out, ++i) {
        *out = this->op(*in,*param);
      }
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryScaleOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryScaleOp (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::Scale>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinarySumOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinarySumOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::Sum>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryProdOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryProdOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::Prod>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryLogOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryLogOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::Log>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryExpOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryExpOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::Exp>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryLowPassOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryLowPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::LowPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryHighPassOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryHighPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::HighPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryBandPassFixedLowerOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryBandPassFixedLowerOp (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::BandPassFixedLower>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
BinaryBandPassFixedUpperOp<EvalT, Traits, InScalarT, FieldScalarT>::
BinaryBandPassFixedUpperOp (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InScalarT,FieldScalarT,BinaryOps::BandPassFixedUpper>(p,dl) {}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT, template<typename,typename> class TernaryOperation>
SimpleTernaryOperation<EvalT, Traits, InScalarT, FieldScalarT, TernaryOperation>::
SimpleTernaryOperation (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InScalarT,
                      typename Albany::StrongestScalarType<InScalarT,
                                                           FieldScalarT>::type,
                      TernaryOperation<InScalarT,FieldScalarT>> (p,dl)
{
  auto pf_layout = Albany::getLayout(p.get<std::string>("Parameter Field Layout"),*dl);
  const auto pf1_name = p.get<std::string>("Parameter Field 1 Name");
  const auto pf2_name = p.get<std::string>("Parameter Field 2 Name");
  param_field1 = PHX::MDField<const FieldScalarT> (pf1_name,pf_layout);
  param_field2 = PHX::MDField<const FieldScalarT> (pf2_name,pf_layout);
  this->addDependentField(param_field1);
  this->addDependentField(param_field2);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT, template<typename,typename> class TernaryOperation>
void SimpleTernaryOperation<EvalT, Traits, InScalarT, FieldScalarT, TernaryOperation>::
evaluateFields (typename Traits::EvalData /* workset */)
{
  PHAL::MDFieldIterator<const InScalarT> in(this->field_in);
  PHAL::MDFieldIterator<const FieldScalarT> param1(param_field1);
  PHAL::MDFieldIterator<const FieldScalarT> param2(param_field2);
  PHAL::MDFieldIterator<OutScalarT> out(this->field_out);
  for (; !in.done(); ++in, ++out, ++param1, ++param2) {
    *out = this->op(*in,*param1,*param2);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
TernaryBandPassOp<EvalT, Traits, InScalarT, FieldScalarT>::
TernaryBandPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleTernaryOperation<EvalT,Traits,InScalarT,FieldScalarT,TernaryOps::BandPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InScalarT, typename FieldScalarT>
Teuchos::RCP<PHX::Evaluator<Traits> >
buildSimpleEvaluatorImpl(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl_in)
{
  Teuchos::RCP<PHX::Evaluator<Traits> > ev;

  const std::string type = p.get<std::string>("Type");
  Teuchos::RCP<PHX::DataLayout> inout_layout;
  Teuchos::RCP<Albany::Layouts> dl;
  if (p.isParameter("Side Set Name")) {
    const std::string ss_name = p.get<std::string>("Side Set Name");
    dl = dl_in->side_layouts.at(ss_name);
  } else {
    dl = dl_in;
  }

  if (type=="Unary Scale") {
    ev = Teuchos::rcp(new UnaryScaleOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Unary Log") {
    ev = Teuchos::rcp(new UnaryLogOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Unary Exp") {
    ev = Teuchos::rcp(new UnaryExpOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Unary Low Pass") {
    ev = Teuchos::rcp(new UnaryLowPassOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Unary High Pass") {
    ev = Teuchos::rcp(new UnaryHighPassOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Unary Band Pass") {
    ev = Teuchos::rcp(new UnaryBandPassOp<EvalT,Traits,InScalarT>(p,dl) );
  } else if (type=="Binary Scale") {
    ev = Teuchos::rcp(new BinaryScaleOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Sum") {
    ev = Teuchos::rcp(new BinarySumOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Prod") {
    ev = Teuchos::rcp(new BinaryProdOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Log") {
    ev = Teuchos::rcp(new BinaryLogOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Exp") {
    ev = Teuchos::rcp(new BinaryExpOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary High Pass") {
    ev = Teuchos::rcp(new BinaryHighPassOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Low Pass") {
    ev = Teuchos::rcp(new BinaryLogOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Band Pass Fixed Lower") {
    ev = Teuchos::rcp(new BinaryBandPassFixedLowerOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Binary Band Pass Fixed Upper") {
    ev = Teuchos::rcp(new BinaryBandPassFixedUpperOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  } else if (type=="Ternary Band Pass") {
    ev = Teuchos::rcp(new TernaryBandPassOp<EvalT,Traits,InScalarT,FieldScalarT>(p,dl) );
  }

  return ev;
}

template<typename EvalT, typename Traits>
Teuchos::RCP<PHX::Evaluator<Traits> >
buildSimpleEvaluator(const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl)
{
  using ST = typename EvalT::ScalarT;
  using PST = typename EvalT::ParamScalarT;
  using MST = typename EvalT::MeshScalarT;
  using RST = RealType;

  Teuchos::RCP<PHX::Evaluator<Traits>> ev;
  const std::string inST = p.get<std::string>("Input Field Scalar Type");
  std::string paramST = inST;
  if (p.isParameter("Parameter Field Scalar Type")) {
    paramST = p.get<std::string>("Parameter Field Scalar Type");
  }

  if (inST=="Scalar") {
    if (paramST=="Scalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,ST,ST>(p,dl);
    } else if (paramST=="ParamScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,ST,PST>(p,dl);
    } else if (paramST=="MeshScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,ST,MST>(p,dl);
    } else if (paramST=="Real") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,ST,RST>(p,dl);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Invalid value for 'Parameter Field Scalar Type' (" + paramST + ").\n");
    }
  } else if (inST=="ParamScalar") {
    if (paramST=="Scalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,PST,ST>(p,dl);
    } else if (paramST=="ParamScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,PST,PST>(p,dl);
    } else if (paramST=="MeshScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,PST,MST>(p,dl);
    } else if (paramST=="Real") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,PST,RST>(p,dl);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Invalid value for 'Parameter Field Scalar Type' (" + paramST + ").\n");
    }
  } else if (inST=="MeshScalar") {
    if (paramST=="Scalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,MST,ST>(p,dl);
    } else if (paramST=="ParamScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,MST,PST>(p,dl);
    } else if (paramST=="MeshScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,MST,MST>(p,dl);
    } else if (paramST=="Real") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,MST,RST>(p,dl);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Invalid value for 'Parameter Field Scalar Type' (" + paramST + ").\n");
    }
  } else if (inST=="Real") {
    if (paramST=="Scalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,RST,ST>(p,dl);
    } else if (paramST=="ParamScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,RST,PST>(p,dl);
    } else if (paramST=="MeshScalar") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,RST,MST>(p,dl);
    } else if (paramST=="Real") {
      ev = buildSimpleEvaluatorImpl<EvalT,Traits,RST,RST>(p,dl);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Invalid value for 'Parameter Field Scalar Type' (" + paramST + ").\n");
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                "Error! Invalid value for 'Input Field Scalar Type' (" + inST + ").\n");
  }
  return ev;
}

} // Namespace LandIce
