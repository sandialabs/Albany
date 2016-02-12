//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficientGradient<EvalT, Traits>::
BasalFrictionCoefficientGradient (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl) :
  grad_beta (p.get<std::string> ("Basal Friction Coefficient Gradient Name"), dl->side_qp_gradient)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = (beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "From File");

  numSideQPs = dl->side_qp_gradient->dimension(2);
  sideDim    = dl->side_qp_gradient->dimension(3);

  basalSideName = p.get<std::string>("Side Set Name");
  if (betaType == "Given Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant and uniform beta, loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
  }
  else if (betaType == "Given Field")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant beta, loaded from file.\n";
#endif
    beta_type = GIVEN_FIELD;

    beta_given = PHX::MDField<ParamScalarT,Cell,Side,Node>(p.get<std::string> ("Given Beta Variable Name"), dl->side_node_scalar);
    GradBF     = PHX::MDField<MeshScalarT,Cell,Side,Node,QuadPoint,Dim>(p.get<std::string> ("Gradient BF Side Variable Name"), dl->side_node_qp_gradient);

    this->addDependentField (beta_given);
    this->addDependentField (GradBF);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->side_node_qp_gradient->dimensions(dims);
    numSideNodes = dims[2];
  }

  this->addEvaluatedField(grad_beta);

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    coordVec = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficientGradient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(grad_beta,fm);

  if (beta_type==GIVEN_FIELD)
  {
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(beta_given,fm);
  }

  if (use_stereographic_map)
    this->utils.setFieldData(coordVec,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (beta_type==POWER_LAW || beta_type==REGULARIZED_COULOMB, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in FELIX::BasalFrictionCoefficientGradient:  \"" << (beta_type==POWER_LAW ? "Power Law" : "Regularized Coulomb")
                << "\" does not allow for gradient calculation\n");

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

    switch (beta_type)
    {
      case GIVEN_CONSTANT:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          for (int dim=0; dim<sideDim; ++dim)
            grad_beta(cell,side,qp,dim) = 0.;
        }
        break;

      case GIVEN_FIELD:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          for (int dim=0; dim<sideDim; ++dim)
          {
            grad_beta(cell,side,qp,dim) = 0.;
            for (int node=0; node<numSideNodes; ++node)
            {
              grad_beta(cell,side,qp,dim) += GradBF(cell,side,node,qp,dim)*beta_given(cell,side,node);
            }
          }
        }
        break;
    }

    // Correct the value if we are using a stereographic map
    if (use_stereographic_map)
    {
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        MeshScalarT x = coordVec(cell,side,qp,0) - x_0;
        MeshScalarT y = coordVec(cell,side,qp,1) - y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        for (int dim=0; dim<sideDim; ++dim)
          grad_beta(cell,side,qp,dim) *= h*h;
      }
    }
  }
}

} // Namespace FELIX
