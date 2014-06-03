//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>
#include <algorithm>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace Aeras {
using Teuchos::rcp;
using PHX::MDALayout;


template<typename EvalT, typename Traits>
Atmosphere_Moisture<EvalT, Traits>::
Atmosphere_Moisture(Teuchos::ParameterList& p,
           const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF             (p.get<std::string> ("Weighted BF Name"),      dl->node_qp_scalar),
  coordVec        (p.get<std::string> ("Coordinate Vector Name"),dl->node_3vector ),
  Velx            (p.get<std::string> ("U Variable Name"),       dl->qp_vector),
  Temp            (p.get<std::string> ("T Variable Name"),       dl->qp_vector),
  VelxResid       (p.get<std::string> ("U Residual Name"),       dl->node_vector),
  TempResid       (p.get<std::string> ("T Residual Name"),       dl->node_vector),
  tracerNames     (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  tracerResidNames(p.get< Teuchos::ArrayRCP<std::string> >("Tracer Residual Names")),
  namesToResid    (),
  numNodes        (dl->node_scalar             ->dimension(0)),
  numQPs          (dl->node_qp_scalar          ->dimension(2)),
  numDims         (dl->node_qp_gradient        ->dimension(3)),
  numLevels       (dl->node_scalar_level       ->dimension(2))
{  
  Teuchos::ArrayRCP<std::string> RequiredTracers;
  RequiredTracers[0] = "Vapor";
  RequiredTracers[1] = "Rain";
  RequiredTracers[2] = "Snow";
  for (int i=0; i<3; ++i) {
    bool found = false;
    for (int j=0; j<3 && !found; ++j)
      if (RequiredTracers[i] == tracerNames[j]) found = true;
    TEUCHOS_TEST_FOR_EXCEPTION(!found, std::logic_error,
      "Aeras::Atmosphere_Moisture requires Vapor, Rain and Snow tracers.");
  }

  this->addDependentField(wBF);
  this->addDependentField(coordVec);
  this->addDependentField(Velx);
  this->addEvaluatedField(Temp);
  this->addDependentField(VelxResid);
  this->addEvaluatedField(TempResid);

  for (int i = 0; i < tracerNames.size(); ++i) {
    namesToResid[tracerNames[i]] = tracerResidNames[i];
    PHX::MDField<ScalarT,Cell,Node> in   (tracerNames[i],       dl->node_scalar_level);
    PHX::MDField<ScalarT,Cell,Node> resid(tracerResidNames[i],  dl->qp_scalar_level);
    TracerIn[tracerNames[i]]         = in;
    TracerResid[tracerResidNames[i]] = resid;
    this->addEvaluatedField(TracerIn   [tracerNames[i]]);
    this->addEvaluatedField(TracerResid[tracerResidNames[i]]);
  }
  this->setName("Aeras::Atmosphere_Moisture"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void Atmosphere_Moisture<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF     ,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(Velx,    fm);
  this->utils.setFieldData(Temp,    fm);
  this->utils.setFieldData(VelxResid,fm);
  this->utils.setFieldData(TempResid,fm);

  for (int i = 0; i < TracerIn.size();    ++i) this->utils.setFieldData(TracerIn[tracerNames[i]], fm);
  for (int i = 0; i < TracerResid.size(); ++i) this->utils.setFieldData(TracerResid[tracerResidNames[i]],fm);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void Atmosphere_Moisture<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  //Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for (int i=0; i < VelxResid.size(); ++i) VelxResid(i)=0.0;
  for (int i=0; i < TempResid.size(); ++i) TempResid(i)=0.0;
  for (int t=0; t < TracerResid.size(); ++t)  
    for (int i=0; i < TracerResid[tracerResidNames[t]].size(); ++i) TracerResid[tracerResidNames[t]](i)=0.0;

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node = 0; node < numNodes; ++node) {
        for (int l=0; l < numLevels; ++l) { 
          TracerResid[namesToResid["Vapor"]](cell,node,l) 
             += 0 * TracerIn["Vapor"](cell,node,l) * wBF(cell,node,qp);
          TracerResid[namesToResid["Rain"]](cell,node,l) 
             += 0 * TracerIn["Rain"](cell,node,l) * wBF(cell,node,qp);
          TracerResid[namesToResid["Snow"]](cell,node,l) 
             += 0 * TracerIn["Snow"](cell,node,l) * wBF(cell,node,qp);
        }
      }
    }
  }

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          VelxResid(cell,node,level) += 0 * Velx(cell,qp,level) * wBF(cell,node,qp);
          TempResid(cell,node,level) += 0 * Temp(cell,qp,level) * wBF(cell,node,qp);
        }
      }
    }
  }

}
}
