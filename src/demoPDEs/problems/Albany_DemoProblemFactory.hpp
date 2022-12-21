//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DEMO_PROBLEM_FACTORY_HPP
#define ALBANY_DEMO_PROBLEM_FACTORY_HPP

#include "Albany_ProblemFactory.hpp"

namespace Albany {

// A concrete problem factory class for basic albany problems
class DemoProblemFactory : public ProblemFactory
{
public:
  obj_ptr_type create (const std::string& key,
                       const Teuchos::RCP<const Teuchos_Comm>& comm,
                       const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
                       const Teuchos::RCP<ParamLib>& paramLib) const;

  bool provides (const std::string& key) const;

  static DemoProblemFactory& instance () {
    static DemoProblemFactory factory;
    return factory;
  }
protected:

  //! Default constructor
  DemoProblemFactory () = default;
};

} // namespace Albany

#endif // ALBANY_DEMO_PROBLEM_FACTORY_HPP
