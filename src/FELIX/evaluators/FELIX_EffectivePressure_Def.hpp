//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "FELIX_HomotopyParameter.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
EffectivePressure<EvalT, Traits>::EffectivePressure (const Teuchos::ParameterList& p,
                                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  N          (p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar),
  alphaField ("Hydraulic-Over-Hydrostatic Potential Ratio",dl->shared_param)
{
  stokes = p.isParameter("Stokes") ? p.get<bool>("Stokes") : false;
  surrogate = stokes && (p.isParameter("Surrogate") ? p.get<bool>("Surrogate") : true);

  TEUCHOS_TEST_FOR_EXCEPTION (surrogate && !stokes, std::logic_error, "Error! Surrogate effective pressure makes sense only for Stokes.\n");

  if (stokes)
  {
    basalSideName = p.get<std::string>("Side Set Name");
    numNodes = dl->node_scalar->dimension(2);
  }
  else
  {
    numNodes = dl->node_scalar->dimension(1);
  }

  if (surrogate)
  {
    this->addDependentField (alphaField);

    regularized = p.get<Teuchos::ParameterList*>("Parameter List")->get("Regularize With Continuation",false);
    printedAlpha = -1.0;
  }
  else
  {
    z_s  = PHX::MDField<ParamScalarT>(p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar);
    phi  = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), dl->node_scalar);

    this->addDependentField (phi);
    this->addDependentField (z_s);
  }

  H   = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar);
  this->addDependentField (H);

  this->addEvaluatedField (N);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density",910);
  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(alphaField,fm);
  this->utils.setFieldData(H,fm);
  if (!surrogate)
  {
    this->utils.setFieldData(z_s,fm);
    this->utils.setFieldData(phi,fm);
  }
  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT alpha = alphaField(0);
//  TEUCHOS_TEST_FOR_EXCEPTION (surrogate && (alpha<0 || alpha>1), Teuchos::Exceptions::InvalidParameter,
//                              "Error! 'Hydraulic-Over-Hydrostatic Potential Ratio' must be in [0,1].\n");
  if (regularized)
  {
    ScalarT homotopyParam = FELIX::HomotopyParameter<EvalT>::value;
    alpha = alpha*std::sqrt(homotopyParam);
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (surrogate && std::fabs(printedAlpha-alpha)>0.0001)
  {
    *output << "[Effective Pressure<" << PHX::typeAsString<EvalT>() << ">]] alpha = " << alpha << "\n";
    printedAlpha = alpha;
  }
#endif
  if (stokes)
  {
    const Albany::SideSetList& ssList = *(workset.sideSets);
    Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

    if (it_ss==ssList.end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
    std::vector<Albany::SideStruct>::const_iterator iter_s;

    if (surrogate)
    {
      for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
      {
        // Get the local data of side and cell
        const int cell = iter_s->elem_LID;
        const int side = iter_s->side_local_id;

        for (int node=0; node<numNodes; ++node)
        {
          // N = p_i-p_w
          // p_i = rho_i*g*H
          // p_w = alpha*p_i
          N (cell,side,node) = (1-alpha)*rho_i*g*H(cell,side,node);
        }
      }
    }
    else
    {
      for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
      {
        // Get the local data of side and cell
        const int cell = iter_s->elem_LID;
        const int side = iter_s->side_local_id;

        for (int node=0; node<numNodes; ++node)
        {
          // N = p_i-p_w
          // p_i = rho_i*g*H
          // p_w = rho_w*g*z_b - phi
          N (cell,side,node) = rho_i*g*H(cell,side,node) + rho_w*g*(z_s(cell,side,node) - H(cell,side,node)) - phi (cell,side,node);
        }
      }
    }
  }
  else
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int node=0; node<numNodes; ++node)
      {
        // N = p_i-p_w
        // p_i = rho_i*g*H
        // p_w = rho_w*g*z_b - phi
        N(cell,node) = rho_i*g*H(cell,node) + rho_w*g*(z_s(cell,node) - H(cell,node) - phi(cell,node));
      }
    }
  }
}

} // Namespace FELIX
