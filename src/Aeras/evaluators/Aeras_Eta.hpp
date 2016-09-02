//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/*
 * Aeras_Eta.hpp
 *
 *  Created on: May 28, 2014
 *      Author: swbova
 */

#ifndef AERAS_ETA_HPP_
#define AERAS_ETA_HPP_

//OG A few comments on eta coordinates:
//Hybrid pressure coords are given by p(eta) = A(eta)*p0 + B(eta)*ps , ps is
//surface pressure. A few restrictions on coefficients A, B, and eta: A+B = eta,
//eta(bottom) = 1, eta(top) = etatop, A(bottom) = 0, B(bottom) = 1,
//A(top) = etatop, so, etatop = ptop/p0 < 1. ptop and p0 are given constants.
//Eta is defined by a monotone function, here the function is
//eta(level) = etatop + (1-etatop)*level/num_of_levels, level=0,..,N.
//Note that momentum, temperature, tracer eqns are solved on mid-levels,
//that is, there are N+1 level interfaces (eta=etatop, eta=1, etc.) and there are N levels' middle
//points. This is why many quantities of interest are calculated on midlevels and
//function eta below has (ScalarT(L)+.5).
//
// tmsmith - A few additional comments (05/24/16)
// This class has been refactored to read A(level+1/2) and B(level+1/2)
// from a file "aeras_eta_coefficients.dat".  If the file
// does not exist, then the coefficients are computed using the original
// formula mentioned above.  The file sould contain numLevels+1 rows and two columns.

#include <fstream>
#include "Kokkos_ViewFactory.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"
#include "Kokkos_DynRankView.hpp"

namespace Aeras {

template<typename EvalT>
class Eta {
public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::DynRankView<ScalarT, PHX::Device> a_kokkos, b_kokkos;
  Kokkos::DynRankView<ScalarT, PHX::Device> A_kokkos, B_kokkos;
  Kokkos::DynRankView<ScalarT, PHX::Device> delta_kokkos;
#endif

  static const Eta<EvalT> &self(const ScalarT ptop=0,
                                const ScalarT p0=0,
                                const int     L=0) {
    static const Eta swc(ptop,p0,L);

    return swc;
  }

  ScalarT     p0() const { return P0;  }
  ScalarT   ptop() const { return Ptop;}
  ScalarT etatop() const { return Etatop;}

  ScalarT   eta(const double L) const { 
    const ScalarT e = Etatop + (1-Etatop)*(ScalarT(L)+.5)/numLevels; 
    return e;
  }

  ScalarT delta(const int L) const { 
    const double etap = L + .5;
    const double etam = L - .5;
    const ScalarT DeltaEta = eta(etap) - eta(etam);
    return  DeltaEta;
  }

  //ScalarT     W(const int level) const { return  (eta(level)-Etatop)/(1-Etatop); }
  //ScalarT     A(const int level) const { return   eta(level)*(1-W(level));       }
  //ScalarT     B(const int level) const { return   eta(level)*   W(level);        }
  ScalarT     A(const int level) const { return  0.5*(a[level]+a[level+1]); }
  ScalarT     B(const int level) const { return  0.5*(b[level]+b[level+1]); }

  //ScalarT     A(const double half_step) const { return  eta(half_step)*(1-(eta(half_step)-Etatop)/(1-Etatop));}
  //ScalarT     B(const double half_step) const { return  eta(half_step)*(   eta(half_step)-Etatop)/(1-Etatop);}
  ScalarT     A(const double half_step) const { int lp = int(half_step+0.5);
                                                return  a[lp]; }
  ScalarT     B(const double half_step) const { int lp = int(half_step+0.5);
                                                return  b[lp]; }

  Eta(const ScalarT ptop, const ScalarT p0, const int L) :
    P0(p0),
    Ptop(ptop),
    Etatop(ptop/p0),
    numLevels(L), 
    a(L+1,0.0),
    b(L+1,0.0)
  {
    std::vector<double> ain(L+1,0.0);
    std::vector<double> bin(L+1,0.0);

    std::ifstream infile("aeras_eta_coefficients.dat");

    if(infile) {

      for (int i=0; i<=L; ++i) {
        infile >> ain[i] >> bin[i];      
        a[i] = (ScalarT) ain[i];
        b[i] = (ScalarT) bin[i];
        //std::cout << "level: " << i << "  " << a[i] << "  " << b[i] << std::endl;
      }
      infile.close();

    } else {
      std::cout << "aeras_eta_coefficients.dat not found, using internal values for a and b! " << std::endl;
      std::cout << "Etatop = " << Etatop << std::endl;
      std::cout << "numLevels = " << L << std::endl;
      // a and b coefficients are defined at 1/2 level itervals: a[0] is at level=1/2 to a[numLevels] numLevels+1/2
      for (int i=0; i<=L; ++i) {
        double half_step = i-0.5;
        a[i] = eta(half_step)*(1-(eta(half_step)-Etatop)/(1-Etatop));
        b[i] = eta(half_step)*(   eta(half_step)-Etatop)/(1-Etatop);

        //std::cout << "i: "          << i << "  " 
        //          << "level: "      << i+1 << "  "
        //          << "i+1/2: "      << half_step+1 << "  " 
        //          << "eta(i+1/2): " << eta(half_step) << "  " 
        //          << "a(i+1/2): "   << a[i] << "  " 
        //          << "b(i+1/2): "   << b[i] << "  " 
        //          << "a+b: "        << a[i]+b[i] << std::endl;
        
      }
    }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    // Convert vector of a and b coefficients to views
    a_kokkos = Kokkos::createDynRankView(a_kokkos, "a", L+1);
    b_kokkos = Kokkos::createDynRankView(b_kokkos, "b", L+1);
    for (int i=0; i<=L; ++i) {
      a_kokkos(i) = a[i];
      b_kokkos(i) = b[i];
    }

    // Compute A and B
    A_kokkos = Kokkos::createDynRankView(A_kokkos, "A", numLevels);
    B_kokkos = Kokkos::createDynRankView(B_kokkos, "B", numLevels);
    for (int level = 0; level < numLevels; level++) {
      A_kokkos(level) = 0.5*(a[level] + a[level+1]);
      B_kokkos(level) = 0.5*(b[level] + b[level+1]);
    }

    // Compute delta eta
    delta_kokkos = Kokkos::createDynRankView(delta_kokkos, "delta", numLevels);
    for (int level = 0; level < numLevels; level++) {
      const double levelp = level + .5;
      const double levelm = level - .5;
      const ScalarT etap = Etatop + (1-Etatop)*(ScalarT(levelp)+.5)/numLevels;
      const ScalarT etam = Etatop + (1-Etatop)*(ScalarT(levelm)+.5)/numLevels;
      delta_kokkos(level) = etap - etam;
    }

#endif
  }

  ~Eta(){}

private:
  const ScalarT P0;
  const ScalarT Ptop;
  const ScalarT Etatop;
  const int     numLevels;
  std::vector<ScalarT> a;
  std::vector<ScalarT> b;
};
}
#endif /* AERAS_ETA_HPP */
