//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  Porosity<EvalT, Traits>::
  Porosity(Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl) :
    porosity(p.get<std::string>("Porosity Name"),dl->qp_scalar),
    is_constant(true),
    isCompressibleSolidPhase(false),
    isCompressibleFluidPhase(false),
    isPoroElastic(false)
  {
    Teuchos::ParameterList* porosity_list = 
      p.get<Teuchos::ParameterList*>("Parameter List");

    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_vector->dimensions(dims);
    numQPs  = dims[1];
    numDims = dims[2];

    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

    std::string type = porosity_list->get("Porosity Type", "Constant");
    if (type == "Constant") {
      is_constant = true;
      // Default value = 0 means no pores in the material
      constant_value = porosity_list->get("Value", 0.0); 
      // Add Porosity as a Sacado-ized parameter
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>
        ("Porosity", this, paramLib);
    }
    else if (type == "Truncated KL Expansion") {
      is_constant = false;
      PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
        fx(p.get<string>("QP Coordinate Vector Name"), dl->qp_vector);
      coordVec = fx;
      this->addDependentField(coordVec);

      exp_rf_kl = 
        Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>
                     (*porosity_list));
      int num_KL = exp_rf_kl->stochasticDimension();

      // Add KL random variables as Sacado-ized parameters
      rv.resize(num_KL);
      for (int i=0; i<num_KL; i++) {
        std::string ss = Albany::strint("Porosity KL Random Variable",i);
        new Sacado::ParameterRegistration<EvalT, SPL_Traits>
          (ss, this, paramLib);
        rv[i] = porosity_list->get(ss, 0.0);
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, 
                                 Teuchos::Exceptions::InvalidParameter,
                                 "Invalid porosity type " << type);
    } 

    // Optional dependence on porePressure and Biot coefficient
    // Switched ON by sending porePressure field in p
    if ( p.isType<string>("Strain Name") ) {

      //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
      //     p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
      //   PHX::MDField<ScalarT,Cell,QuadPoint>
      //     tmp(p.get<string>("QP Pore Pressure Name"), scalar_dl);
      //   porePressure = tmp;
      //  this->addDependentField(porePressure);

      PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>
        ts(p.get<string>("Strain Name"), dl->qp_tensor);
      strain = ts;
      this->addDependentField(strain);

      isPoroElastic = true;
      initialPorosity_value = 
        porosity_list->get("Initial Porosity Value", 0.0);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>
        ("Initial Porosity Value", this, paramLib);
    }
    else {
      // porosity will not change in this case.
      isPoroElastic=false; 
      initialPorosity_value=0.0;
    }

    if ( p.isType<string>("Biot Coefficient Name") ) {
      PHX::MDField<ScalarT,Cell,QuadPoint>
        btp(p.get<string>("Biot Coefficient Name"), dl->qp_scalar);
      biotCoefficient = btp;
      isCompressibleSolidPhase = true;
      isCompressibleFluidPhase = true;
      isPoroElastic = true;
      this->addDependentField(biotCoefficient);
    }

    if ( p.isType<string>("QP Pore Pressure Name") ) {
      PHX::MDField<ScalarT,Cell,QuadPoint>
        ppn(p.get<string>("QP Pore Pressure Name"), dl->qp_scalar);
      porePressure = ppn;
      isCompressibleSolidPhase = true;
      isCompressibleFluidPhase = true;
      isPoroElastic = true;
      this->addDependentField(porePressure);

      // typically Kgrain >> Kskeleton
      GrainBulkModulus = 
        porosity_list->get("Grain Bulk Modulus Value", 10.0e12); 
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>
        ("Grain Bulk Modulus Value", this, paramLib);
    }
    this->addEvaluatedField(porosity);
    this->setName("Porosity"+PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Porosity<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(porosity,fm);
    if (!is_constant) this->utils.setFieldData(coordVec,fm);
    if (isPoroElastic) this->utils.setFieldData(strain,fm);
    if (isCompressibleSolidPhase) this->utils.setFieldData(biotCoefficient,fm);
    if (isCompressibleFluidPhase) this->utils.setFieldData(porePressure,fm);

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Porosity<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    std::size_t numCells = workset.numCells;

    if (is_constant) {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  porosity(cell,qp) = constant_value;
        }
      }
    }
    else {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          Teuchos::Array<MeshScalarT> point(numDims);
          for (std::size_t i=0; i<numDims; i++)
            point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
          porosity(cell,qp) = exp_rf_kl->evaluate(point, rv);
        }
      }
    }
    if ((isPoroElastic) && (isCompressibleSolidPhase) && (isCompressibleFluidPhase)) {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  porosity(cell,qp) = initialPorosity_value;

    	  Teuchos::Array<MeshScalarT> point(numDims);
          for (std::size_t i=0; i<numDims; i++) {
            porosity(cell,qp) += biotCoefficient(cell,qp)*strain(cell,qp,i,i)
              + porePressure(cell,qp)
              *(biotCoefficient(cell,qp)-initialPorosity_value)/GrainBulkModulus;
          }
          // // for debug
          // std::cout << "initial Porosity: " << initialPorosity_value << endl;
          // std::cout << "Pore Pressure: " << porePressure << endl;
          // std::cout << "Biot Coefficient: " << biotCoefficient << endl;
          // std::cout << "Grain Bulk Modulus " << GrainBulkModulus << endl;

          // porosity(cell,qp) += (1.0 - initialPorosity_value)
          //   /GrainBulkModulus*porePressure(cell,qp);
    	  // // for large deformation, \phi = J \dot \phi_{o}
        }
      }
    } else {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          porosity(cell,qp) = initialPorosity_value;

          Teuchos::Array<MeshScalarT> point(numDims);
          for (std::size_t i=0; i<numDims; i++) {
            porosity(cell,qp) += strain(cell,qp,i,i);
          }
        }
      }
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT,typename Traits>
  typename Porosity<EvalT,Traits>::ScalarT&
  Porosity<EvalT,Traits>::getValue(const std::string &n)
  {
    if (n == "Porosity")
      return constant_value;
    else if (n == "Initial Porosity Value")
      return initialPorosity_value;
    else if (n == "Grain Bulk Modulus Value")
      return GrainBulkModulus;
    for (int i=0; i<rv.size(); i++) {
      if (n == Albany::strint("Porosity KL Random Variable",i))
        return rv[i];
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error! Logic error in getting parameter " << n
                               << " in Porosity::getValue()" << std::endl);
    return constant_value;
  }

  //----------------------------------------------------------------------------
}

