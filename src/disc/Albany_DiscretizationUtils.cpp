#include "Albany_DiscretizationUtils.hpp"

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C2_FEM.hpp"

namespace Albany {

int computeWorksetSize(const int worksetSizeMax,
                       const int ebSizeMax)
{
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
    // compute numWorksets, and shrink workset size to minimize padding
    const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
    return (1 + (ebSizeMax-1) / numWorksets);
  }
}

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis (const CellTopologyData& cell_topo,
                   const FE_Type fe_type, const int order)
{
  using namespace Intrepid2;
  Teuchos::RCP<Basis<PHX::Device, RealType, RealType> > basis;
  std::string topo = cell_topo.name;
  topo = topo.substr(0,topo.find("_"));
  switch (fe_type) {
    case FE_Type::HGRAD:
      TEUCHOS_TEST_FOR_EXCEPTION (order<0, std::logic_error,
          "Error! Order of FE space cannot be negative.\n");

      if (topo=="Line") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_LINE_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Triangle") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Quadrilateral") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Tetrahedron") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Hexahedron") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Wedge") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_WEDGE_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_WEDGE_C2_FEM<PHX::Device>() );
        else
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
              "Error! Wedge supported only for order=1,2. Requested: " + std::to_string(order) + "\n");
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Unrecognized/unsupported topology for HGRAD: " + topo + "\n");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (false, std::runtime_error,
          "FE Type " + e2str(fe_type) + " not yet supported.\n");
  }

  return basis;
}
// inline Teuchos::RCP<const panzer::FieldPattern>
// createFieldPattern (const FE_Type fe_type,
//                     const shards::CellTopology& cell_topo)
// {
//   Teuchos::RCP<const panzer::FieldPattern> fp;
//   switch (fe_type) {
//     case FE_Type::P1:
//       fp = Teuchos::rcp(new panzer::NodalFieldPattern(cell_topo));
//       break;
//     case FE_Type::P0:
//       fp = Teuchos::rcp(new panzer::ElemFieldPattern(cell_topo));
//       break;
//     default:
//       TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
//           "Error! Unsupported FE_Type.\n");
//   }
//   return fp;
// }


Teuchos::RCP<Thyra_MultiVector>
readScalarFileSerial (const std::string& fname,
                      const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                      const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  // It's a scalar, so we already know MultiVector has only 1 vector
  auto mvec = Thyra::createMembers(vs,1);

  if (comm->getRank() != 0) {
    // Only process 0 will load the file...
    return mvec;
  }

  auto nonConstData = getNonconstLocalData(mvec->col(0));

  std::ifstream ifile;
  ifile.open(fname.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error,
        "[readScalarFileSerial] Error! Unable to open the file.\n"
        "  - file name: " << fname << "\n");

  GO numNodes;
  ifile >> numNodes;
  TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstData.size(),
      Teuchos::Exceptions::InvalidParameterValue,
      "[readScalarFileSerial] Error! Unexpected number of nodes.\n"
      "  - file name: " << fname << "\n"
      "  - number of nodes in file: " << numNodes << "\n"
      "  - expected number of nodes: " << nonConstData.size() << "\n");

  for (GO i = 0; i < numNodes; i++) {
    ifile >> nonConstData[i];
  }

  ifile.close();

  return mvec;
}

Teuchos::RCP<Thyra_MultiVector>
readVectorFileSerial (const std::string& fname,
                      const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                      const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  int numComponents;
  GO numNodes;
  std::ifstream ifile;
  if (comm->getRank() == 0) {
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error,
          "[readVectorFileSerial] Error! Unable to open the file.\n"
          "  - file name: " << fname << "\n");

    ifile >> numNodes >> numComponents;
  }

  Teuchos::broadcast(*comm,0,1,&numComponents);
  auto mvec = Thyra::createMembers(vs,numComponents);

  if (comm->getRank()==0) {
    auto nonConstData = getNonconstLocalData(mvec);
    TEUCHOS_TEST_FOR_EXCEPTION (numComponents != nonConstData.size(),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readVectorFileSerial] Error! Unexpected number of components.\n"
        "  - file name: " << fname << "\n"
        "  - number of components in file: " << numComponents << "\n"
        "  - expected number of components: " << nonConstData.size() << "\n");

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstData[0].size(),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readVectorFileSerial] Error! Unexpected number of nodes.\n"
        "  - file name: " << fname << "\n"
        "  - number of nodes in file: " << numNodes << "\n"
        "  - expected number of nodes: " << nonConstData[0].size() << "\n");

    for (int icomp=0; icomp<numComponents; ++icomp) {
      auto comp_view = nonConstData[icomp];
      for (GO i=0; i<numNodes; ++i)
        ifile >> comp_view[i];
    }
    ifile.close();
  }

  return mvec;
}

Teuchos::RCP<Thyra_MultiVector>
readLayeredScalarFileSerial (const std::string &fname,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  size_t numLayers=0;
  GO numNodes;

  std::ifstream ifile;
  if (comm->getRank()==0) {
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error,
        "[readLayeredScalarFileSerial] Error! Unable to open the file.\n"
        "  - file name: " << fname << "\n");

    ifile >> numNodes >> numLayers;
  }

  Teuchos::broadcast(*comm,0,1,&numLayers);
  auto mvec = Thyra::createMembers(vs,numLayers);

  if (comm->getRank()==0) {
    auto nonConstData = getNonconstLocalData(mvec);
    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstData[0].size(),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readLayeredScalarFileSerial] Error! Unexpected number of nodes.\n"
        "  - file name: " << fname << "\n"
        "  - number of nodes in file: " << numNodes << "\n"
        "  - expected number of nodes: " << nonConstData[0].size() << "\n");
    TEUCHOS_TEST_FOR_EXCEPTION (numLayers != normalizedLayersCoords.size(),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readLayeredScalarFileSerial] Error! Unexpected number of layers.\n"
        "  - file name: " << fname << "\n"
        "  - number of layers in file: " << numLayers << "\n"
        "  - expected number of layers: " << normalizedLayersCoords.size() << "\n"
        " To fix this, please specify the correct layered data dimension when you register the state.\n");

    for (size_t il = 0; il < numLayers; ++il) {
      ifile >> normalizedLayersCoords[il];
    }

    for (size_t il=0; il<numLayers; ++il) {
      for (GO i=0; i<numNodes; ++i) {
        ifile >> nonConstData[il][i];
      }
    }
    ifile.close();
  }

  return mvec;
}


Teuchos::RCP<Thyra_MultiVector>
readLayeredVectorFileSerial (const std::string &fname,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             std::vector<double>& normalizedLayersCoords,
                             const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  int numVectors=0;
  int numLayers,numComponents;
  GO numNodes;
  std::ifstream ifile;
  if (comm->getRank()==0) {
    ifile.open(fname.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION (!ifile.is_open(), std::runtime_error,
        "[readLayeredVectorFileSerial] Error! Unable to open the file.\n"
        "  - file name: " << fname << "\n");

    ifile >> numNodes >> numComponents >> numLayers;
    numVectors = numLayers*numComponents;
  }

  Teuchos::broadcast(*comm,0,1,&numVectors);
  auto mvec = Thyra::createMembers(vs,numVectors);

  if (comm->getRank()==0) {
    auto nonConstData = getNonconstLocalData(mvec);

    TEUCHOS_TEST_FOR_EXCEPTION (numNodes != nonConstData[0].size(),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readLayeredVectorFileSerial] Error! Unexpected number of nodes.\n"
        "  - file name: " << fname << "\n"
        "  - number of nodes in file: " << numNodes << "\n"
        "  - expected number of nodes: " << nonConstData[0].size() << "\n");
    TEUCHOS_TEST_FOR_EXCEPTION (numLayers != static_cast<int>(normalizedLayersCoords.size()),
        Teuchos::Exceptions::InvalidParameterValue,
        "[readLayeredVectorFileSerial] Error! Unexpected number of layers.\n"
        "  - file name: " << fname << "\n"
        "  - number of layers in file: " << numLayers << "\n"
        "  - expected number of layers: " << normalizedLayersCoords.size() << "\n"
        " To fix this, please specify the correct layered data dimension when you register the state.\n");

    normalizedLayersCoords.resize(numLayers);
    for (int il=0; il<numLayers; ++il) {
      ifile >> normalizedLayersCoords[il];
    }

    // Layer ordering: before switching component, we first do all the layers of the current component
    // This is because with the stk field (natural ordering) we want to keep the layer dimension last.
    // Ex: a 2D field f(i,j) would be stored at the raw array position i*num_cols+j. In our case,
    //     num_cols is the number of layers, and num_rows the number of field components
    for (int il=0; il<numLayers; ++il) {
      for (int icomp(0); icomp<numComponents; ++icomp) {
        Teuchos::ArrayRCP<ST> col_vals = nonConstData[icomp*numLayers+il];
        for (GO i=0; i<numNodes; ++i) {
          ifile >> col_vals[i];
        }
      }
    }
    ifile.close();
  }

  return mvec;
}

Teuchos::RCP<Thyra_MultiVector>
loadField (const std::string& field_name,
           const Teuchos::ParameterList& field_params,
           const CombineAndScatterManager& cas_manager,
           const Teuchos::RCP<const Teuchos_Comm>& comm,
           bool node, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out,
           std::vector<double>& norm_layers_coords)
{
  // Getting the serial and (possibly) parallel vector spaces
  auto serial_vs = cas_manager.getOwnedVectorSpace();
  auto vs        = cas_manager.getOverlappedVectorSpace();

  std::string field_type = (node ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");

  // The serial service multivector
  Teuchos::RCP<Thyra_MultiVector> serial_req_mvec;

  std::string fname = field_params.get<std::string>("File Name");

  *out << "  - Reading " << field_type << " field '" << field_name << "' from file '" << fname << "' ... ";
  out->getOStream()->flush();
  // Read the input file and stuff it in the Tpetra multivector

  if (scalar) {
    if (layered) {
      serial_req_mvec = readLayeredScalarFileSerial (fname,cas_manager.getOwnedVectorSpace(),norm_layers_coords,comm);

      // Broadcast the normalized layers coordinates
      int size = norm_layers_coords.size();
      Teuchos::broadcast(*comm,0,size,norm_layers_coords.data());
    } else {
      serial_req_mvec = readScalarFileSerial (fname,cas_manager.getOwnedVectorSpace(),comm);
    }
  } else {
    if (layered) {
      serial_req_mvec = readLayeredVectorFileSerial (fname,cas_manager.getOwnedVectorSpace(),norm_layers_coords,comm);

      // Broadcast the normalized layers coordinates
      int size = norm_layers_coords.size();
      Teuchos::broadcast(*comm,0,size,norm_layers_coords.data());
    } else {
      serial_req_mvec = readVectorFileSerial (fname,cas_manager.getOwnedVectorSpace(),comm);
    }
  }
  *out << "done!\n";

  if (field_params.isParameter("Scale Factor")) {
    Teuchos::Array<double> scale_factors;
    if (field_params.isType<Teuchos::Array<double>>("Scale Factor")) {
      scale_factors = field_params.get<Teuchos::Array<double> >("Scale Factor");
      TEUCHOS_TEST_FOR_EXCEPTION (scale_factors.size()!=static_cast<int>(serial_req_mvec->domain()->dim()),
                                  Teuchos::Exceptions::InvalidParameter,
                                  "Error! The given scale factors vector size does not match the field dimension.\n");
    } else if (field_params.isType<double>("Scale Factor")) {
      scale_factors.resize(serial_req_mvec->domain()->dim());
      std::fill_n(scale_factors.begin(),scale_factors.size(),field_params.get<double>("Scale Factor"));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                 "Error! Invalid type for parameter 'Scale Factor'. Should be either 'double' or 'Array(double)'.\n");
    }

    *out << "   - Scaling " << field_type << " field '" << field_name << "' with scaling factors [" << scale_factors[0];
    for (int i=1; i<scale_factors.size(); ++i) {
      *out << " " << scale_factors[i];
    }
    *out << "]\n";

    for (int i=0; i<scale_factors.size(); ++i) {
      serial_req_mvec->col(i)->scale (scale_factors[i]);
    }
  }

  // Fill the (possibly) parallel vector
  auto field_mv = Thyra::createMembers(vs,serial_req_mvec->domain()->dim());
  cas_manager.scatter(*serial_req_mvec, *field_mv, CombineMode::INSERT);
  return field_mv;
}

Teuchos::RCP<Thyra_MultiVector>
fillField (const std::string& field_name,
           const Teuchos::ParameterList& field_params,
           const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
           bool nodal, bool scalar, bool layered,
           const Teuchos::RCP<Teuchos::FancyOStream> out,
           std::vector<double>& norm_layers_coords)
{
  std::string temp_str;
  std::string field_type = (nodal ? "Node" : "Elem");
  field_type += (layered ? " Layered" : "");
  field_type += (scalar ? " Scalar" : " Vector");

  Teuchos::RCP<Thyra_MultiVector> field_mv;
  if (field_params.isParameter("Random Value")) {
    *out << "  - Filling " << field_type << " field '" << field_name << "' with random values.\n";

    Teuchos::Array<std::string> randomize = field_params.get<Teuchos::Array<std::string> >("Random Value");
    field_mv = Thyra::createMembers(entities_vs,randomize.size());

    if (layered) {
      *out << "    - Filling layers normalized coordinates linearly in [0,1].\n";

      int size = norm_layers_coords.size();
      if (size==1) {
        norm_layers_coords[0] = 1.;
      } else {
        int n_int = size-1;
        double dx = 1./n_int;
        norm_layers_coords[0] = 0.;
        for (int i=0; i<n_int; ++i) {
          norm_layers_coords[i+1] = norm_layers_coords[i]+dx;
        }
      }
    }

    // If there are components that were marked to not be randomized,
    // we look for the parameter 'Field Value' and use the corresponding entry.
    // If there is no such parameter, we fill the non random entries with zeroes.
    Teuchos::Array<double> values;
    if (field_params.isParameter("Field Value")) {
      values = field_params.get<Teuchos::Array<double> >("Field Value");
    } else {
      values.resize(randomize.size(),0.);
    }

    for (int iv=0; iv<randomize.size(); ++iv) {
      if (randomize[iv]=="false" || randomize[iv]=="no") {
        *out << "    - Using constant value " << values[iv] << " for component " << iv << ", which was marked as not random.\n";
        field_mv->col(iv)->assign(values[iv]);
      }
    }
  } else if (field_params.isParameter("Field Value")) {
    Teuchos::Array<double> values;
    if (field_params.isType<Teuchos::Array<double>>("Field Value")) {
      values = field_params.get<Teuchos::Array<double> >("Field Value");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()==0 , Teuchos::Exceptions::InvalidParameter,
          "Error! The given field value array has size 0.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()==1 && !scalar , Teuchos::Exceptions::InvalidParameter,
          "Error! The given field value array has size 1, but the field is not scalar.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (values.size()>1 && scalar , Teuchos::Exceptions::InvalidParameter,
          "Error! The given field value array has size >1, but the field is scalar.\n");
    } else if (field_params.isType<double>("Field Value")) {
      if (scalar) {
        values.resize(1);
        values[0] = field_params.get<double>("Field Value");
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (!field_params.isParameter("Vector Dim"), std::logic_error,
            "Error! Cannot determine dimension of " << field_type << " field '" << field_name << "'. "
            "In order to fill with constant value, either specify 'Vector Dim', or make 'Field Value' an Array(double).\n");
        values.resize(field_params.get<int>("Vector Dim"));
        std::fill_n(values.begin(),values.size(),field_params.get<double>("Field Value"));
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
            "Error! Invalid type for parameter 'Field Value'. Should be either 'double' or 'Array(double)'.\n");
    }

    if (layered) {
      *out << "  - Filling " << field_type << " field '" << field_name << "' with constant value " << values << " and filling layers normalized coordinates linearly in [0,1].\n";

      int size = norm_layers_coords.size();
      if (size==1) {
        norm_layers_coords[0] = 1.;
      } else {
        int n_int = size-1;
        double dx = 1./n_int;
        norm_layers_coords[0] = 0.;
        for (int i=0; i<n_int; ++i) {
          norm_layers_coords[i+1] = norm_layers_coords[i]+dx;
        }
      }
    } else {
      *out << "  - Filling " << field_type << " field '" << field_name << "' with constant value " << values << ".\n";
    }

    field_mv = Thyra::createMembers(entities_vs,values.size());
    for (int iv(0); iv<field_mv->domain()->dim(); ++iv) {
      field_mv->col(iv)->assign(values[iv]);
    }
  }

  return field_mv;
}

} // namespace Albany
