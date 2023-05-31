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
      const Teuchos::RCP<const Teuchos_Comm>& comm);

    //! Destructor
    ~DiscretizationFactory() {}

    static Teuchos::RCP<AbstractMeshStruct>
    createMeshStruct (Teuchos::RCP<Teuchos::ParameterList> disc_params,
                      Teuchos::RCP<const Teuchos_Comm> comm, const int numParams);

    Teuchos::RCP<AbstractMeshStruct> getMeshStruct() {
      return meshStruct;
    }

    Teuchos::RCP<MeshSpecs> createMeshSpecs();

    Teuchos::RCP<MeshSpecs> createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh);

    Teuchos::RCP<AbstractDiscretization>
    createDiscretization(unsigned int num_equations,
                         const Teuchos::RCP<StateInfoStruct>& sis,
                         const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null);


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

    void
    setMeshStructBulkData(
      const Teuchos::RCP<StateInfoStruct>& sis);

   void
    setMeshStructBulkData(
      const Teuchos::RCP<StateInfoStruct>& sis,
      const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis);

    Teuchos::RCP<AbstractDiscretization> createDiscretizationFromInternalMeshStruct(
      const int neq,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes);

    Teuchos::RCP<AbstractDiscretization> createDiscretizationFromInternalMeshStruct(
      const int neq,
      const std::map<int,std::vector<std::string> >& sideSetEquations,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes);
    
    /* This function overwrite previous discretization parameter list */
    void
    setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params);
    
    private:

    //! Private to prohibit copying
    DiscretizationFactory(const DiscretizationFactory&);

    //! Private to prohibit copying
    DiscretizationFactory& operator=(const DiscretizationFactory&);

    const std::map<int,std::vector<std::string> > empty_side_set_equations;
    const std::map<std::string,Teuchos::RCP<StateInfoStruct> > empty_side_set_sis;

    void
    setFieldData(Teuchos::RCP<AbstractDiscretization> disc,
                                        const Teuchos::RCP<StateInfoStruct>& sis);

    void completeDiscSetup(Teuchos::RCP<AbstractDiscretization> disc);

  protected:

    //! Parameter list specifying what element to create
    Teuchos::RCP<Teuchos::ParameterList> discParams;

    //! Parameter list specifying solver parameters
    Teuchos::RCP<Teuchos::ParameterList> piroParams;

    Teuchos::RCP<const Teuchos_Comm> comm;

    //The following are for Aeras hydrostatic problems
    int numLevels;
    int numTracers;
    
    Teuchos::RCP<AbstractMeshStruct> meshStruct;

    //Number of parameters
    int num_params = 0;
};

} // namespace Albany

#endif // ALBANY_DISCRETIZATIONFACTORY_HPP
