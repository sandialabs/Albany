//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "PHAL_Dirichlet.hpp"
#include "Albany_ThyraUtils.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template<typename EvalT,typename Traits>
DirichletBase<EvalT, Traits>::
DirichletBase(Teuchos::ParameterList& p) :
  offset(p.get<int>("Equation Offset")),
  nodeSetID(p.get<std::string>("Node Set ID"))
{
  value = p.get<RealType>("Dirichlet Value");

  std::string name = p.get< std::string >("Dirichlet Name");
  const Teuchos::RCP<PHX::DataLayout> dummy = p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");
  const PHX::Tag<ScalarT> fieldTag(name, dummy);

  this->addEvaluatedField(fieldTag);

  this->setName(name+PHX::print<EvalT>());

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
               ("Parameter Library", Teuchos::null);

  this->registerSacadoParameter(name, paramLib);

  {
    // Impose an ordering on DBC evaluators. This code works with the function
    // imposeOrder defined elsewhere. It happens that "Data Layout" is Dummy, so
    // we can use it.
    if (p.isType<std::string>("BCOrder Dependency")) {
      PHX::Tag<ScalarT> order_depends_on(p.get<std::string>("BCOrder Dependency"), dummy);
      this->addDependentField(order_depends_on);
    }
    if (p.isType<std::string>("BCOrder Evaluates")) {
      PHX::Tag<ScalarT> order_evaluates(p.get<std::string>("BCOrder Evaluates"), dummy);
      this->addEvaluatedField(order_evaluates);
    }
  }
}

template<typename EvalT, typename Traits>
void DirichletBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Residual, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      // (*f)[lunk] = ((*x)[lunk] - this->value);
      f_nonconstView[lunk] = x_constView[lunk] - this->value;
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    x_constView     = Albany::getLocalData(x);
    f_nonconstView  = Albany::getNonconstLocalData(f);
  }

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];
    index[0] = lunk;

    // Extract the row, zero it out, then put j_coeff on diagonal
    Albany::getLocalRowValues(jac,lunk,matrixIndices,matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, lunk, matrixIndices(), matrixEntries());
    Albany::setLocalRowValues(jac, lunk, index(), value());

    if (fillResid) {
      f_nonconstView[lunk] = x_constView[lunk] - this->value.val();
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<const Thyra_Vector>       x  = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>             f  = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    x_constView    = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }
  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (dirichletWorkset.f != Teuchos::null) {
      f_nonconstView[lunk] = x_constView[lunk] - this->value.val();
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JV_nonconst2dView[i][lunk] = j_coeff*Vx_const2dView[i][lunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fp_nonconst2dView[i][lunk] = -this->value.dx(dirichletWorkset.param_offset+i);
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;

  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);

    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lunk] = 0.0;
        Vp_nonconst2dView[col][lunk] = 0.0;
       }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lunk] = 0.0;
        fpV_nonconst2dView[col][lunk] = 0.0;
      }
    }
  }
}

// **********************************************************************
// Simple evaluator to aggregate all Dirichlet BCs into one "field"
// **********************************************************************

template<typename EvalT, typename Traits>
DirichletAggregator<EvalT, Traits>::
DirichletAggregator(Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> dl =  p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  const std::vector<std::string>& dbcs = *p.get<Teuchos::RCP<std::vector<std::string> > >("DBC Names");

  for (unsigned int i=0; i<dbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(dbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("DBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Dirichlet Aggregator"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DirichletAggregator<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
}

