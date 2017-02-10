//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
NSMaterialProperty<EvalT, Traits>::
NSMaterialProperty(Teuchos::ParameterList& p) :
  name_mp(p.get<std::string>("Material Property Name")),
  layout(p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout")),
  matprop(name_mp,layout),
  rank(layout->rank()),
  dims(),
  matPropType(SCALAR_CONSTANT)
{
  layout->dimensions(dims);

  double default_value = p.get("Default Value", 1.0);

  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  Teuchos::ParameterList* mp_list =
    p.get<Teuchos::ParameterList*>("Parameter List");
  std::string type = mp_list->get("Type", "Constant");

  if (type == "Constant") {
    if (rank == 2) {
      matPropType = SCALAR_CONSTANT;
      scalar_constant_value = mp_list->get("Value", default_value);

      // Add property as a Sacado-ized parameter
      this->registerSacadoParameter(name_mp, paramLib);
    }
    else if (rank == 3) {
      matPropType = VECTOR_CONSTANT;
      PHX::index_size_type numDims = dims[2];
      Teuchos::Array<double> tmp =
      mp_list->get< Teuchos::Array<double> >("Value");
      vector_constant_value.resize(numDims);
      TEUCHOS_TEST_FOR_EXCEPTION(vector_constant_value.size() != numDims,
			 std::logic_error,
			 "Vector constant value for material property " <<
			 name_mp << " has size " <<
			 vector_constant_value.size() << " but expected size "
			 << numDims);

      for (PHX::index_size_type i=0; i<numDims; i++)
	vector_constant_value[i] = tmp[i];

      // Add property as a Sacado-ized parameter
      for (PHX::DataLayout::size_type i=0; i<numDims; i++)
        this->registerSacadoParameter(Albany::strint(name_mp,i), paramLib);
    }
    else if (rank == 4) {
      matPropType = TENSOR_CONSTANT;
      PHX::index_size_type numRows = dims[2];
      PHX::index_size_type numCols = dims[3];
      Teuchos::TwoDArray<double> tmp =
	mp_list->get< Teuchos::TwoDArray<double> >("Value");
      TEUCHOS_TEST_FOR_EXCEPTION(tensor_constant_value.getNumRows() != numRows ||
			 tensor_constant_value.getNumCols() != numCols,
			 std::logic_error,
			 "Tensor constant value for material property " <<
			 name_mp << " has dimensions " <<
			 tensor_constant_value.getNumRows() << "x" <<
			 tensor_constant_value.getNumCols() <<
			 " but expected dimensions " <<
			 numRows << "x" << numCols);
      tensor_constant_value = Teuchos::TwoDArray<ScalarT>(numRows, numCols);
      for (PHX::index_size_type i=0; i<numRows; i++)
	for (PHX::index_size_type j=0; j<numCols; j++)
	  tensor_constant_value(i,j) = tmp(i,j);

      // Add property as a Sacado-ized parameter
      for (PHX::DataLayout::size_type i=0; i<numRows; i++)
	for (PHX::DataLayout::size_type j=0; j<numCols; j++)
          this->registerSacadoParameter(Albany::strint(Albany::strint(name_mp,i),j), paramLib);
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			 "Invalid material property rank " << rank <<
			 ".  Acceptable values are 2 (scalar), " <<
			 "3 (vector), or 4 (tensor)");
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion" ||
	   type == "Log Normal RF" ||
	   type == "Exponential Truncated KL Expansion") {
    if (type == "Truncated KL Expansion")
      matPropType = KL_RAND_FIELD;
    else if (type == "Log Normal RF" ||
	     type == "Exponential Truncated KL Expansion")
      matPropType = EXP_KL_RAND_FIELD;

    Teuchos::RCP<PHX::DataLayout> coord_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("Coordinate Vector Data Layout");
    coordVec = PHX::MDField<MeshScalarT>(
      p.get<std::string>("Coordinate Vector Name"),
      coord_dl);
    this->addDependentField(coordVec);
    std::vector<PHX::DataLayout::size_type> coord_dims;
    coord_dl->dimensions(coord_dims);
    point.resize(coord_dims[2]);

    exp_rf_kl =
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*mp_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint(name_mp + " KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = mp_list->get(ss, 0.0);
    }
  }
#endif
  else if (type == "SQRT Temperature Dependent") {
    matPropType = SQRT_TEMP;
    scalar_constant_value = mp_list->get("Reference Value", default_value);
    ref_temp = mp_list->get("Reference Temperature", default_value);
    T = PHX::MDField<ScalarT>(
      p.get<std::string>("Temperature Variable Name"),
      layout);
    this->addDependentField(T);

    // Add property as a Sacado-ized parameter
    this->registerSacadoParameter(name_mp+" Reference Value", paramLib);
  }
  else if (type == "invSQRT Temperature Dependent") {
    matPropType = INV_SQRT_TEMP;
    scalar_constant_value = mp_list->get("Reference Value", default_value);
    ref_temp = mp_list->get("Reference Temperature", default_value);
    T = PHX::MDField<ScalarT>(
      p.get<std::string>("Temperature Variable Name"),
      layout);
    this->addDependentField(T);

    // Add property as a Sacado-ized parameter
    this->registerSacadoParameter(name_mp+" Reference Value", paramLib);
  }
  else if (type == "Transport Mean Free Path") {
    matPropType = NEUTRON_DIFFUSION;
    sigma_a = PHX::MDField<ScalarT>(
      p.get<std::string>("Absorption Cross Section Name"),
      layout);
    sigma_s = PHX::MDField<ScalarT>(
      p.get<std::string>("Scattering Cross Section Name"),
      layout);
    mu = PHX::MDField<ScalarT>(
      p.get<std::string>("Average Scattering Angle Name"),
      layout);
    this->addDependentField(sigma_a);
    this->addDependentField(sigma_s);
    this->addDependentField(mu);
  }
  else if (type == "Time Dependent") {
    matPropType = TIME_DEP_SCALAR;
    timeValues = mp_list->get<Teuchos::Array<RealType>>("Time Values").toVector();
    depValues = mp_list->get<Teuchos::Array<RealType>>("Dependent Values").toVector();

    TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == depValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"Dependent Values\" do not match" );

      // Add property as a Sacado-ized parameter
    this->registerSacadoParameter(name_mp, paramLib);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid material property type " << type);
  }

  this->addEvaluatedField(matprop);
  this->setName(name_mp);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void NSMaterialProperty<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(matprop,fm);
#ifdef ALBANY_STOKHOS
  if (matPropType == KL_RAND_FIELD || matPropType == EXP_KL_RAND_FIELD)
    this->utils.setFieldData(coordVec,fm);
#endif
  if (matPropType == SQRT_TEMP || matPropType == INV_SQRT_TEMP)
    this->utils.setFieldData(T,fm);
  if (matPropType == NEUTRON_DIFFUSION) {
    this->utils.setFieldData(sigma_a,fm);
    this->utils.setFieldData(sigma_s,fm);
    this->utils.setFieldData(mu,fm);
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void NSMaterialProperty<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (matPropType == SCALAR_CONSTANT) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
        matprop(cell,qp) = scalar_constant_value;
      }
    }
  }
  else if (matPropType == VECTOR_CONSTANT) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	for (std::size_t dim=0; dim < dims[2]; ++dim) {
	  matprop(cell,qp,dim) = vector_constant_value[dim];
	}
      }
    }
  }
  else if (matPropType == TENSOR_CONSTANT) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	for (std::size_t dim1=0; dim1 < dims[2]; ++dim1) {
	  for (std::size_t dim2=0; dim2 < dims[3]; ++dim2) {
	    matprop(cell,qp,dim1,dim2) = tensor_constant_value(dim1,dim2);
	  }
	}
      }
    }
  }
  else if (matPropType == SQRT_TEMP) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	matprop(cell,qp) = scalar_constant_value / sqrt(ref_temp) * sqrt(T(cell,qp));
      }
    }
  }
  else if (matPropType == INV_SQRT_TEMP) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	matprop(cell,qp) = scalar_constant_value * sqrt(ref_temp) / sqrt(T(cell,qp));
      }
    }
  }
  else if (matPropType == NEUTRON_DIFFUSION) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	matprop(cell,qp) =
	  1.0 / (3.0*(1.0 - mu(cell,qp))*(sigma_a(cell,qp) + sigma_s(cell,qp)));
      }
    }
  }
  else if (matPropType == TIME_DEP_SCALAR) {

    RealType time = workset.current_time;
    TEUCHOS_TEST_FOR_EXCEPTION(
       time > timeValues.back(), Teuchos::Exceptions::InvalidParameter,
      "Time is growing unbounded!" );

    RealType slope;
    unsigned int index(0);

    while (timeValues[index] < time)
      index++;

    if (index == 0)
      scalar_constant_value = depValues[index];
    else {
      slope = ((depValues[index] - depValues[index - 1]) /
             (timeValues[index] - timeValues[index - 1]));
      scalar_constant_value = depValues[index-1] + slope * (time - timeValues[index - 1]);
    }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
         matprop(cell,qp) = scalar_constant_value;
      }
    }
  }
#ifdef ALBANY_STOKHOS
  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < dims[1]; ++qp) {
	for (std::size_t i=0; i<point.size(); i++)
	  point[i] =
	    Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
    matprop(cell,qp) = exp_rf_kl->evaluate(point, rv);
    if (matPropType == EXP_KL_RAND_FIELD)
       matprop(cell,qp) = std::exp(matprop(cell,qp));
    }
  }
 }
#endif
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename NSMaterialProperty<EvalT,Traits>::ScalarT&
NSMaterialProperty<EvalT,Traits>::getValue(const std::string &n)
{
  if (matPropType == SCALAR_CONSTANT ||
      matPropType == SQRT_TEMP ||
      matPropType == INV_SQRT_TEMP ||
      matPropType == TIME_DEP_SCALAR) {
    return scalar_constant_value;
  }
  else if (matPropType == VECTOR_CONSTANT) {
    for (std::size_t dim=0; dim<vector_constant_value.size(); ++dim)
      if (n == Albany::strint(name_mp,dim))
	return vector_constant_value[dim];
  }
  else if (matPropType == TENSOR_CONSTANT) {
    for (std::size_t dim1=0; dim1<tensor_constant_value.getNumRows(); ++dim1)
      for (std::size_t dim2=0; dim2<tensor_constant_value.getNumCols(); ++dim2)
	if (n == Albany::strint(Albany::strint(name_mp,dim1),dim2))
	  return tensor_constant_value(dim1,dim2);
  }
#ifdef ALBANY_STOKHOS
  else if (matPropType == KL_RAND_FIELD || matPropType == EXP_KL_RAND_FIELD) {
    for (int i=0; i<rv.size(); i++)
      if (n == Albany::strint(name_mp + " KL Random Variable",i))
	return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting paramter " << n
		     << " in NSMaterialProperty::getValue()" << std::endl);
  return scalar_constant_value;
}

// **********************************************************************
// **********************************************************************
}

