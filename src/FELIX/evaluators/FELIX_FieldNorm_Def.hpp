//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
FieldNorm<EvalT, Traits>::FieldNorm (const Teuchos::ParameterList& p,
                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  field      (p.get<std::string> ("Field Name"), dl->node_vector),
  field_norm (p.get<std::string> ("Field Norm Name"), dl->node_scalar)
{
  this->addDependentField(field);

  this->addEvaluatedField(field_norm);

  homotopyParam = 0;

  if (p.isParameter("Regularization"))
  {
    regularizationParam = p.get<double>("Regularization");
    homotopyParam = &regularizationParam;
  }
  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  numDim   = dims[2];

  this->setName("FieldNorm"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void FieldNorm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(field_norm,fm);
}

template<typename EvalT,typename Traits>
void FieldNorm<EvalT,Traits>::setHomotopyParamPtr(ScalarT* h)
{
  homotopyParam = h;
#ifdef OUTPUT_TO_SCREEN
    printedH = -1234.56789;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void FieldNorm<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

    if (homotopyParam!=0 &&  printedH!=*homotopyParam)
    {
        *output << "[Field Norm] h = " << *homotopyParam << "\n";
        printedH = *homotopyParam;
    }
#endif

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  ScalarT ff = 0;
  if (homotopyParam!=0 && *homotopyParam!=0)
    ff = pow(10.0, -10.0*(*homotopyParam));

  ScalarT norm;
  for (int cell=0; cell<workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      norm = 0;
      for (int dim=0; dim<numDim; ++dim)
        norm += std::pow(field(cell,node,dim),2);
      field_norm(cell,node) = std::sqrt (norm + ff);
    }
  }
#else
  Kokkos::parallel_for (workset.numCells, *this);
#endif
}

template<typename EvalT, typename Traits>
void FieldNorm<EvalT, Traits>::operator() (const int i) const
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  ScalarT ff = 0;
  if (homotopyParam!=0 && *homotopyParam!=0)
    ff = pow(10.0, -10.0*(*homotopyParam));

  ScalarT norm;
  for (int node=0; node < numNodes; ++node)
  {
    norm = 0;
    for (int dim=0; dim<numDim; ++dim)
      norm += std::pow(field(i,node,dim),2);
    field_norm(i,node) = std::sqrt (norm + ff);
  }
#else
  Kokkos::parallel_for (workset.numCells, *this);
#endif
}

} // Namespace FELIX
