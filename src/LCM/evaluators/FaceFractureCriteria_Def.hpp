/*
 * FaceFractureCriteria_Def.hpp
 *
 *  Created on: Jul 20, 2012
 *      Author: jrthune
 */

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

  template<typename EvalT, typename Traits>
  FaceFractureCriteria<EvalT, Traits>::
  FaceFractureCriteria(const Teuchos::ParameterList& p) :
    faceAve(p.get<std::string>("Face Average Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("Face Vector Data Layout")),
    yieldStrength(p.get<RealType>("Yield Name")),
    criteriaMet(p.get<std::string>("Criteria Met Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("Face Scalar Data Layout")),
    temp(p.get<std::string>("Temp2 Name"),
            p.get<Teuchos::RCP<PHX::DataLayout> >("Cell Scalar Data Layout"))
  {
      this->addDependentField(faceAve);

      this->addEvaluatedField(criteriaMet);
      this->addEvaluatedField(temp); // temp for testing

      // Get Dimensions
      Teuchos::RCP<PHX::DataLayout> vec_dl =
        p.get<Teuchos::RCP<PHX::DataLayout> >("Face Vector Data Layout");
      std::vector<PHX::DataLayout::size_type> dims;
      vec_dl->dimensions(dims);

      worksetSize = dims[0];
      numFaces    = dims[1];
      numComp     = dims[2];

      this->setName("FaceFractureCriteria"+PHX::TypeString<EvalT>::value);
  }

  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT,Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
          PHX::FieldManager<Traits>& fm)
  {
      this->utils.setFieldData(faceAve, fm);
      this->utils.setFieldData(criteriaMet,fm);

      this->utils.setFieldData(temp,fm);  // temp for testing

  }

  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT,Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
      // test
      criterion = "Test Fracture";

      if (criterion == "Test Fracture"){
          testFracture(faceAve);
      }
      else
          cout << "Invalid Fracture Criterion" << std::endl;

  }

  // Test fracture criterion
  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT,Traits>::testFracture(
          PHX::MDField<ScalarT,Cell,Face,VecDim> faceAve)
  {
      for (std::size_t cell=0; cell < worksetSize; ++cell)
      {
          for (std::size_t face=0; face<numFaces; ++face){
              ScalarT max_comp = 0.0;
              criteriaMet(cell,face) = 0;
              for (std::size_t comp=0; comp<numComp; ++comp){
                  max_comp = std::max(faceAve(cell,face,comp),max_comp);
              }
              if (max_comp > yieldStrength*0.1){
                  criteriaMet(cell,face) = 1;
                  // for debug
                  cout << "Fracture Criteria met for (cell, face): " << cell << ","
                          << face << " Max Stress: " << max_comp << std::endl;
              }
              else
                  cout << "Criteria not met for (cell,face)" << cell << "," << face
                       << " Max Stress: " << max_comp << std::endl;
          }

          // hack to force evaluation
          temp(cell) = 0.0;
      }

  }

  // Traction based criterion
  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT,Traits>::tractionCriterion(
          PHX::MDField<ScalarT,CEll,Face,VecDim> faceAve)
  {

  }

}  // namespace LCM

