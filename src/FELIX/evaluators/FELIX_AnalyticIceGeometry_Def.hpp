//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
AnalyticIceGeometry<EvalT, Traits>::AnalyticIceGeometry (const Teuchos::ParameterList& p,
                                                         const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec  (p.get<std::string> ("Coordinate QP Vector Name"), dl->qp_gradient ),
  H         (p.get<std::string> ("Ice Thickness QP Variable Name"), dl->qp_scalar),
  z_s       (p.get<std::string> ("Surface Height QP Variable Name"), dl->qp_scalar)
{
  this->addDependentField(coordVec);

  this->addEvaluatedField(z_s);
  this->addEvaluatedField(H);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQp  = dims[1];
  numDim = dims[2];

  rho = p.get<Teuchos::ParameterList*>("Physical Parameters")->get<double>("Ice Density",910.0);
  g = p.get<Teuchos::ParameterList*>("Physical Parameters")->get<double>("Gravity Acceleration",9.8);
  L = p.get<Teuchos::ParameterList*>("Hydrology Parameters")->get<double>("Domain Length",1.0);
  dx = p.get<Teuchos::ParameterList*>("Hydrology Parameters")->get<double>("Domain dx",1.0);

  this->setName("AnalyticIceGeometry"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AnalyticIceGeometry<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(H,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AnalyticIceGeometry<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQp; ++qp)
    {
      MeshScalarT x = coordVec(cell,qp,0);
      H (cell,qp) = z_s(cell,qp) = (L -3*dx - x) * 0.0111 /(rho*g);
    }
  }
}

} // Namespace FELIX
