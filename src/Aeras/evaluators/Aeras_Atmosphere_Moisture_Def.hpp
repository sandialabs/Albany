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
  wBF             (p.get<std::string> ("Weighted BF Name"),              dl->node_qp_scalar),
  coordVec        (p.get<std::string> ("QP Coordinate Vector Name"),     dl->qp_vector     ),
  Velx            (p.get<std::string> ("QP Velx"),                       dl->qp_scalar_level),
  Temp            (p.get<std::string> ("QP Temperature"),                dl->qp_scalar_level),
  TempSrc         (p.get<std::string> ("Temperature Source"),            dl->node_scalar_level),
  tracerNames     (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  tracerSrcNames(p.get< Teuchos::ArrayRCP<std::string> >("Tracer Source Names")),
  namesToSrc      (),
  numNodes        (dl->node_scalar             ->dimension(1)),
  numQPs          (dl->node_qp_scalar          ->dimension(2)),
  numDims         (dl->node_qp_gradient        ->dimension(3)),
  numLevels       (dl->node_scalar_level       ->dimension(2))
{  
  Teuchos::ArrayRCP<std::string> RequiredTracers(3);
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
  this->addEvaluatedField(TempSrc);

  for (int i = 0; i < tracerNames.size(); ++i) {
    namesToSrc[tracerNames[i]] = tracerSrcNames[i];
    PHX::MDField<ScalarT,Cell,Node> in   (tracerNames[i],   dl->qp_scalar_level);
    PHX::MDField<ScalarT,Cell,Node> src(tracerSrcNames[i],  dl->node_scalar_level);
    TracerIn[tracerNames[i]]     = in;
    TracerSrc[tracerSrcNames[i]] = src;
    this->addEvaluatedField(TracerIn   [tracerNames[i]]);
    this->addEvaluatedField(TracerSrc[tracerSrcNames[i]]);
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
  this->utils.setFieldData(TempSrc, fm);

  for (int i = 0; i < TracerIn.size();  ++i) this->utils.setFieldData(TracerIn[tracerNames[i]], fm);
  for (int i = 0; i < TracerSrc.size(); ++i) this->utils.setFieldData(TracerSrc[tracerSrcNames[i]],fm);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void Atmosphere_Moisture<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  //Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for (int i=0; i < TempSrc.size(); ++i) TempSrc(i)=0.0;

  for (int t=0; t < TracerSrc.size(); ++t)  
    for (int i=0; i < TracerSrc[tracerSrcNames[t]].size(); ++i) TracerSrc[tracerSrcNames[t]](i)=0.0;

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node = 0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) { 
          TracerSrc[namesToSrc["Vapor"]](cell,node,level) 
               += 0 * TracerIn["Vapor"] (cell,qp  ,level) * wBF(cell,node,qp);
          TracerSrc[namesToSrc["Rain"]] (cell,node,level) 
               += 0 * TracerIn["Rain"]  (cell,qp  ,level) * wBF(cell,node,qp);
          TracerSrc[namesToSrc["Snow"]] (cell,node,level) 
               += 0 * TracerIn["Snow"]  (cell,qp  ,level) * wBF(cell,node,qp);
        }
      }
    }
  }

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          TempSrc(cell,node,level) += 0 * Temp(cell,qp,level) * wBF(cell,node,qp);
        }
      }
    }
  }

}
}
