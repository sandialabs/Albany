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
    isPoroElastic(false),
    hasStrain(false),
    hasJ(false),
    hasTemp(false)
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
        fx(p.get<std::string>("QP Coordinate Vector Name"), dl->qp_vector);
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
    if ( p.isType<std::string>("Strain Name") ) {

      //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
      //     p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
      //   PHX::MDField<ScalarT,Cell,QuadPoint>
      //     tmp(p.get<std::string>("QP Pore Pressure Name"), scalar_dl);
      //   porePressure = tmp;
      //  this->addDependentField(porePressure);

      hasStrain = true;

      PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>
        ts(p.get<std::string>("Strain Name"), dl->qp_tensor);
      strain = ts;
      this->addDependentField(strain);

      isCompressibleSolidPhase = true;
      isCompressibleFluidPhase = true;
      isPoroElastic = true;
      initialPorosityValue = 
        porosity_list->get("Initial Porosity Value", 0.0);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>
        ("Initial Porosity Value", this, paramLib);
    }
    else if ( p.isType<std::string>("DetDefGrad Name") ) {
      hasJ = true;
      PHX::MDField<ScalarT,Cell,QuadPoint>
        tj(p.get<std::string>("DetDefGrad Name"), dl->qp_scalar);
      J = tj;
      this->addDependentField(J);
      isPoroElastic = true;
      initialPorosityValue = 
        porosity_list->get("Initial Porosity Value", 0.0);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>
        ("Initial Porosity Value", this, paramLib);
    }
    else {
      // porosity will not change in this case.
      isPoroElastic=false; 
      initialPorosityValue=0.0;
    }

    if ( p.isType<std::string>("Biot Coefficient Name") ) {
      PHX::MDField<ScalarT,Cell,QuadPoint>
        btp(p.get<std::string>("Biot Coefficient Name"), dl->qp_scalar);
      biotCoefficient = btp;
      isCompressibleSolidPhase = true;
      isCompressibleFluidPhase = true;
      isPoroElastic = true;
      this->addDependentField(biotCoefficient);
    }

    if ( p.isType<std::string>("QP Pore Pressure Name") ) {
      PHX::MDField<ScalarT,Cell,QuadPoint>
        ppn(p.get<std::string>("QP Pore Pressure Name"), dl->qp_scalar);
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

    if ( p.isType<std::string>("QP Temperature Name") ) {
      PHX::MDField<ScalarT,Cell,QuadPoint>
        ppn(p.get<std::string>("QP Temperature Name"), dl->qp_scalar);
      Temperature = ppn;
      this->addDependentField(Temperature);



         if ( p.isType<std::string>("Skeleton Thermal Expansion Name") ) {
              PHX::MDField<ScalarT,Cell,QuadPoint>
              skte(p.get<std::string>("Skeleton Thermal Expansion Name"), dl->qp_scalar);
              skeletonThermalExpansion = skte;
              this->addDependentField(skeletonThermalExpansion);


            if ( p.isType<std::string>("Reference Temperature Name") ) {
              PHX::MDField<ScalarT,Cell,QuadPoint>
              reftemp(p.get<std::string>("Reference Temperature Name"), dl->qp_scalar);
              refTemperature = reftemp;
              hasTemp = true;
              this->addDependentField(refTemperature);

        }
      }
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
    if (isPoroElastic && hasStrain) this->utils.setFieldData(strain,fm);
    if (isPoroElastic && hasJ) this->utils.setFieldData(J,fm);
    if (isPoroElastic && hasTemp) this->utils.setFieldData(Temperature,fm);
    if (isPoroElastic && hasTemp) this->utils.setFieldData(refTemperature,fm);
    if (isPoroElastic && hasTemp) this->utils.setFieldData(skeletonThermalExpansion,fm);
    if (isCompressibleSolidPhase) this->utils.setFieldData(biotCoefficient,fm);
    if (isCompressibleFluidPhase) this->utils.setFieldData(porePressure,fm);

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Porosity<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    std::size_t numCells = workset.numCells;

    ScalarT temp;

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

    // if the porous media is deforming
    if ((isPoroElastic) && (isCompressibleSolidPhase) && (isCompressibleFluidPhase)) {


      if ( hasStrain ) {
        for (std::size_t cell=0; cell < numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {

            // small deformation; only valid for small porosity changes
            porosity(cell,qp) = initialPorosityValue;

            Teuchos::Array<MeshScalarT> point(numDims);

            for (std::size_t i=0; i<numDims; i++) {
              porosity(cell,qp) = initialPorosityValue + biotCoefficient(cell,qp)*strain(cell,qp,i,i)
                + porePressure(cell,qp)
                *(biotCoefficient(cell,qp)-initialPorosityValue)/GrainBulkModulus;
            }
    	    // Set Warning message
    	    if ( porosity(cell,qp) < 0 ) {
              std::cout << "negative porosity detected. Error! \n";
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
      } else if ( hasJ )
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          if (hasTemp == false){
        	porosity(cell,qp) = initialPorosityValue*std::exp(
        			            GrainBulkModulus/(porePressure(cell,qp) + GrainBulkModulus)*
        			            biotCoefficient(cell,qp)*std::log(J(cell,qp)) +
        		                biotCoefficient(cell,qp)/(porePressure(cell,qp) + GrainBulkModulus)*
        		                porePressure(cell,qp));
          } else{
           	temp = 1.0 + porePressure(cell,qp)/GrainBulkModulus
           			- 3.0*skeletonThermalExpansion(cell,qp)*
           			(Temperature(cell,qp)-refTemperature(cell,qp));

        	porosity(cell,qp) = initialPorosityValue*std::exp(
		                        biotCoefficient(cell,qp)*std::log(J(cell,qp)) +
	                            biotCoefficient(cell,qp)/GrainBulkModulus*porePressure(cell,qp)-
	                            3.0*J(cell,qp)*skeletonThermalExpansion(cell,qp)*
	                            (Temperature(cell,qp)-refTemperature(cell,qp))/temp);
        }


  	    // Set Warning message
  	    if ( porosity(cell,qp) < 0 ) {
              std::cout << "negative porosity detected. Error! \n";
  	    }

        }
      }        
    } else {
      for (std::size_t cell=0; cell < numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          porosity(cell,qp) = initialPorosityValue;
          /*
          if ( hasStrain ) {
            Teuchos::Array<MeshScalarT> point(numDims);
            for (std::size_t i=0; i<numDims; i++) {
              porosity(cell,qp) += strain(cell,qp,i,i);
            }
          }
          */
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
      return initialPorosityValue;
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

