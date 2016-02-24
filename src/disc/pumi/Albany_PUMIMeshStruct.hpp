//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_PUMIMESHSTRUCT_HPP
#define ALBANY_PUMIMESHSTRUCT_HPP

#include "Albany_APFMeshStruct.hpp"

namespace Albany {

class PUMIMeshStruct : public APFMeshStruct {

  public:

    PUMIMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT);

    ~PUMIMeshStruct();

    msType meshSpecsType();

    // For RC: Recoverable.
    void setMesh(apf::Mesh2* new_mesh);

    virtual apf::Field* createNodalField(char const* name, int valueType);

private:

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    void buildBoxMesh(int nex, int ney, int nez,
        double wx, double wy, double wz, bool is);

};

}
#endif
