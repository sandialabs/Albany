//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
LoadStateField<EvalT, Traits>::
LoadStateField(const Teuchos::ParameterList& p) 
{  
  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");

  PHX::MDField<ScalarT> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("State Field Layout") );
  data = f;

  this->addEvaluatedField(data);
  this->setName("Load State Field" );
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void LoadStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data,fm);
}

// **********************************************************************
template<typename ScalarT>
void load (const Albany::MDArray& mda, PHX::MDField<ScalarT>& f) {
#define loop(i, dim) for (int i = 0; i < mda.dimension(dim); ++i)
  switch (f.rank()) {
  case 1:
    loop(i, 0)
      f(i) = mda(i);
    break;
  case 2:
    loop(i, 0) loop(j, 1)
      f(i, j) = mda(i, j);
    break;
  case 3:
    loop(i, 0) loop(j, 1) loop(k, 2)
      f(i, j, k) = mda(i, j, k);
    break;
  case 4:
    loop(i, 0) loop(j, 1) loop(k, 2) loop(l, 3)
      f(i, j, k, l) = mda(i, j, k, l);
    break;
  case 5:
    loop(i, 0) loop(j, 1) loop(k, 2) loop(l, 3) loop(m, 3)
      f(i, j, k, l, m) = mda(i, j, k, l, m);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "dims.size() \notin {1,2,3,4,5}.");
  }
#undef loop
}

template<typename EvalT, typename Traits>
void LoadStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  //cout << "LoadStateField importing state " << stateName << " to field " 
  //     << fieldName << " with size " << data.size() << endl;

  load((*workset.stateArrayPtr)[stateName], data);
}

// **********************************************************************
}

