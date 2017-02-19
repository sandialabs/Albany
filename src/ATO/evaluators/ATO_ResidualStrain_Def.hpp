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

namespace ATO {

template<typename EvalT, typename Traits>
ResidualStrain<EvalT, Traits>::
ResidualStrain(Teuchos::ParameterList& p) :
  strain  (p.get<std::string>("QP Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout"))
{
  Teuchos::ParameterList* residStrainParams = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  coordVec = decltype(coordVec)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  this->addDependentField(coordVec);

  if(residStrainParams->isSublist("Shear XY")){
    Teuchos::ParameterList& plist = residStrainParams->sublist("Shear XY");
    shear_xy_kl = Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(plist));
    int num_KL = shear_xy_kl->stochasticDimension();

    shear_xy_rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("KL Random Variable",i);
      shear_xy_rv[i] = plist.get(ss, 0.0);
    }
  }
  if(residStrainParams->isSublist("Shear XZ")){
    Teuchos::ParameterList& plist = residStrainParams->sublist("Shear XZ");
    shear_xz_kl = Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(plist));
    int num_KL = shear_xz_kl->stochasticDimension();

    shear_xz_rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("KL Random Variable",i);
      shear_xz_rv[i] = plist.get(ss, 0.0);
    }
  }
  if(residStrainParams->isSublist("Shear YZ")){
    Teuchos::ParameterList& plist = residStrainParams->sublist("Shear YZ");
    shear_yz_kl = Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(plist));
    int num_KL = shear_yz_kl->stochasticDimension();

    shear_yz_rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("KL Random Variable",i);
      shear_yz_rv[i] = plist.get(ss, 0.0);
    }
  }
  if(residStrainParams->isSublist("Volumetric")){
    Teuchos::ParameterList& plist = residStrainParams->sublist("Volumetric");
    vol_strn_kl = Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(plist));
    int num_KL = vol_strn_kl->stochasticDimension();

    vol_strn_rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("KL Random Variable",i);
      vol_strn_rv[i] = plist.get(ss, 0.0);
    }
  }

  this->addEvaluatedField(strain);
  this->setName("Residual Strain"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ResidualStrain<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(strain,fm);
  this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ResidualStrain<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  switch (numDims) {
  case 2:
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        Teuchos::Array<MeshScalarT> point(numDims);
        for(int i=0; i<numDims; i++)
          point[i] = coordVec(cell,qp,i);
        ScalarT shear_xy = shear_xy_kl->evaluate(point, shear_xy_rv);
        ScalarT vol_strn = vol_strn_kl->evaluate(point, vol_strn_rv);

        strain(cell,qp,0,0) = vol_strn/2.0;
        strain(cell,qp,1,1) = vol_strn/2.0;
        strain(cell,qp,0,1) = shear_xy;
        strain(cell,qp,1,0) = shear_xy;
      }
    }
    break;
  case 3:
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        Teuchos::Array<MeshScalarT> point(numDims);
        for(int i=0; i<numDims; i++)
          point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
        ScalarT shear_xy = shear_xy_kl->evaluate(point, shear_xy_rv);
        ScalarT shear_xz = shear_xz_kl->evaluate(point, shear_xz_rv);
        ScalarT shear_yz = shear_yz_kl->evaluate(point, shear_yz_rv);
        ScalarT vol_strn = vol_strn_kl->evaluate(point, vol_strn_rv);
  
        strain(cell,qp,0,0) = vol_strn/3.0;
        strain(cell,qp,1,1) = vol_strn/3.0;
        strain(cell,qp,2,2) = vol_strn/3.0;
        strain(cell,qp,0,1) = shear_xy;
        strain(cell,qp,1,0) = shear_xy;
        strain(cell,qp,0,2) = shear_xz;
        strain(cell,qp,2,0) = shear_xz;
        strain(cell,qp,1,2) = shear_yz;
        strain(cell,qp,2,1) = shear_yz;
      }
    }
    break;
  }
}

}

