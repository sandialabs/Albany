//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_PROBLEM_FACTORY_HPP
#define LANDICE_PROBLEM_FACTORY_HPP

#include "Albany_ProblemFactory.hpp"

namespace LandIce
{

  /*!
   * \brief A factory class to instantiate LandIce objects
   */

class LandIceProblemFactory : public Albany::ProblemFactory {
public:
  obj_ptr_type create (const std::string& key,
                       const Teuchos::RCP<const Teuchos_Comm>&     comm,
                       const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
                       const Teuchos::RCP<ParamLib>&               paramLib) const;

  bool provides (const std::string& key) const;

  static LandIceProblemFactory& instance () {
    static LandIceProblemFactory factory;
    return factory;
  }
private:
  LandIceProblemFactory () = default;
};

} // Namespace LandIce

#endif // LANDICE_PROBLEM_FACTORY_HPP

