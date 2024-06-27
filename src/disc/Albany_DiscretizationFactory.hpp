//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_DISCRETIZATIONFACTORY_HPP
#define ALBANY_DISCRETIZATIONFACTORY_HPP

#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractMeshStruct.hpp"

#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractDiscretization objects
 */
class DiscretizationFactory {
  public:

    //! Default constructor
    DiscretizationFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
      const Teuchos::RCP<const Teuchos_Comm>& comm,
      const bool explicit_scheme_ = false);

    //! Destructor
    ~DiscretizationFactory() {}

    Teuchos::RCP<AbstractMeshStruct>
    createMeshStruct (Teuchos::RCP<Teuchos::ParameterList> disc_params,
                      Teuchos::RCP<const Teuchos_Comm> comm, const int numParams);

    Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > createMeshSpecs();

    Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh);



    Teuchos::RCP<AbstractDiscretization>
    createDiscretization(unsigned int num_equations,
                         const std::map<int,std::vector<std::string> >& sideSetEquations,
                         const Teuchos::RCP<StateInfoStruct>& sis,
                         const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis,
                         const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null);

    void
    setMeshStructFieldData(
      const Teuchos::RCP<StateInfoStruct>& sis);

    void
    setMeshStructFieldData(
      const Teuchos::RCP<StateInfoStruct>& sis,
      const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis);

    void setMeshStructBulkData();

    /* This function overwrite previous discretization parameter list */
    void
    setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params);
    
  protected:
    Teuchos::RCP<AbstractDiscretization> createDiscretizationFromMeshStruct(
      const Teuchos::RCP<AbstractMeshStruct>& mesh,
      const int neq,
      const std::map<int,std::vector<std::string> >& sideSetEquations,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes);

    const std::map<int,std::vector<std::string> > empty_side_set_equations;
    const std::map<std::string,Teuchos::RCP<StateInfoStruct> > empty_side_set_sis;

    //! Parameter list specifying what element to create
    Teuchos::RCP<Teuchos::ParameterList> discParams;

    //! Parameter list specifying solver parameters
    Teuchos::RCP<Teuchos::ParameterList> piroParams;

    Teuchos::RCP<const Teuchos_Comm> comm;

    //The following are for Aeras hydrostatic problems
    int numLevels;
    int numTracers;
    
    //Flag for explicit time-integration scheme, used in Aeras
    bool explicit_scheme;

    Teuchos::RCP<AbstractMeshStruct> meshStruct;

    //Number of parameters
    int num_params = 0;
};

} // namespace Albany

#endif // ALBANY_DISCRETIZATIONFACTORY_HPP
