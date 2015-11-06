//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Phalanx_DataLayout.hpp>

namespace LCM
{

//
//
//
template<typename EvalT, typename Traits>
VariationalModel<EvalT, Traits>::
VariationalModel(
    Teuchos::ParameterList * p,
    Teuchos::RCP<Albany::Layouts> const & dl)
{
  return;
}

} // namespace LCM

