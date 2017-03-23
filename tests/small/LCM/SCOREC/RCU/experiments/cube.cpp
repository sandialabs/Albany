/* This program generates an.yaml file based on a number of optional inputs.

   Build with the command
       g++ cube.cpp

   The simplest command is this:
       ./a.out input
   It produces input.yaml, which is the input to AlbanyT.
     Appending options produces various inputs. For example, to create the input
   for the stretched cube with RCU on, run
       ./a.out input elastic bctensile adapt uniform notransform rcu
     The following options are implemented. If one is not provided on the
   command line, the opposite behavior occurs. Not all options work together.
       adapt: Use MeshAdapt.
       fine: Double the resolution.
       test: Halve the resolution.
       uniform: Use Unif Size instead of SPR Size.
       elastic: Run with linear elasticity instead of J2.
       project: Use L2 projection instead of PUMI's QP data transfer.
       bctensile: Stretch the cube instead of using Glen's original cube BCs.
       bccon: Like Glen's BCs, except keep the top and bottom parallel.
       quad: Use second-order tets.
       notransform: Do not use Lie algebra transformations.
       rcu: Update the reference configuration.
       parallel: Run with four ranks instead of one.
*/

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "me.yaml.hpp"

#define prp(a) std::cout << std::setw(20) << #a << " " << a << std::endl;
#define check(a) else if (v == #a) a = true

void.yaml () {
  bool
    adapt = 0,
    refine0 = 0,
    resize0 = 0,
    fine = 0,
    uniform = 0,
    elastic = 0,
    mechanic = 1,
    bccon = 0, bcrot = 0, bctensile = 0, bcsqueeze = 0,
    project = 0,
    quad = 0,
    notransform = 0,
    test = 0,
    rcu = 0,
    parallel = 0;
  double step_factor = 1;
  int bc_type;
  for (int i = 0; i < me.yaml::argc; ++i) {
    const std::string v = me.yaml::argv[i];
    if (false) ;
    check(adapt);
    check(refine0);
    check(resize0);
    check(fine);
    check(uniform);
    check(elastic);
    check(project);
    check(bccon);
    check(bcrot);
    check(bctensile);
    check(bcsqueeze);
    check(quad);
    check(notransform);
    check(rcu);
    check(test);
    check(parallel);
    else if (v == "stepfactor") {
      step_factor = atof(me.yaml::argv[i+1]);
      ++i;
    }
  }
  if (refine0) { adapt = true; uniform = true; resize0 = true; }
  if (bcrot) bc_type = 2; else if (bccon) bc_type = 1;
  else if (bctensile) bc_type = 3; else if (bcsqueeze) bc_type = 4;
  else bc_type = 0;
  const int max_steps = test ? 80 : 400;
  const int nadapt = refine0 ? 0 : 2*max_steps;
  const double step_size = 0.1*step_factor;
  const int write_interval = std::max(1, (int) std::ceil(1/step_factor));
  const double target_element_size = (test ? 2 : fine ? 0.5 : 1)*0.1;
  std::cout << "\n";
  prp(elastic);
  prp(mechanic);
  prp(bc_type);
  std::cout << "\n";
  prp(adapt);
  prp(uniform);
  prp(refine0);
  prp(resize0);
  prp(fine);
  prp(nadapt);
  prp(quad);
  std::cout << "\n";
  prp(rcu);
  prp(project);
  prp(notransform);
  std::cout << "\n";
  prp(test);
  prp(parallel);
  std::string outfn, fn_prefix; {
    std::stringstream ss;
    ss << "out_cube_"
       << "e" << elastic
       << "m" << mechanic
       << "b" << bc_type
       << "a" << adapt
       << "u" << (refine0 ? "o" : "") << (fine ? "f" : "") << uniform
       << "q" << quad
       << "R" << rcu
       << "pr" << project
       << "tr" << ! notransform
       << "te" << test;
    fn_prefix = ss.str() + "_";
    outfn = ss.str() + ".vtk";
  }

  { pl a;
    { pl a("Problem");
      if (mechanic) {
        p("Name", "Mechanics 3D");
        p("MaterialDB Filename",
          elastic ? "materials_elastic.yaml" : "materials_mechanical.yaml");
      } else if (elastic) {
        p("Name", "Elasticity 3D");
        { pl a("Elastic Modulus");
          p("Elastic Modulus Type", "Constant");
          p("Value", 100.0);
        } // Elastic Modulus
        { pl a("Poissons Ratio");
          p("Poissons Ratio Type", "Constant");
          p("Value", 0.29);
        } // Poissons Ratio
      } else {
        std::cerr << "Invalid mechanic and elastic settings.\n";
        return;
      }
      p("Solution Method", "Continuation");
      p("Phalanx Graph Visualization Detail", 2);
      { pl a("Dirichlet BCs");
        p("DBC on NS ns_1 for DOF X", 0.0);
        p("DBC on NS ns_1 for DOF Y", 0.0);
        p("DBC on NS ns_1 for DOF Z", 0.0);
        switch (bc_type) {
        case 1:
          // Constrain + ...
          p("DBC on NS ns_4 for DOF X", 0.0);
          p("DBC on NS ns_4 for DOF Y", 0.0);
          // fall through
        case 0:
          // Shear.
          { pl a("Time Dependent DBC on NS ns_4 for DOF Z");
            p("Time Values", "Array(double)", "{0.0, 0.2, 8.0}");
            p("BC Values", "Array(double)", "{0.0, 0.0, 4.0}");
          } // Time Dependent DBC on NS ns_4 for DOF Z
          break;
        case 2:
          // Shear + flatten.
          p("DBC on NS ns_4 for DOF Y", 0.0);
          { pl a("Time Dependent DBC on NS ns_4 for DOF X");
            p("Time Values", "Array(double)",
              "{0, 0.200, 0.400, 0.800, 1.200, 1.600, 2.000, 2.400, 2.800, 3.200, 3.600, 4.000}");
            p("BC Values", "Array(double)",
              "{0, 0.000, -0.218, -0.426, -0.613, -0.771, -0.891, -0.969, -1.000, -0.982, -0.918, -0.809}");
          } // Time Dependent DBC on NS ns_4 for DOF X
          { pl a("Time Dependent DBC on NS ns_4 for DOF Z");
            p("Time Values", "Array(double)",
              "{0, 0.200, 0.400, 0.800, 1.200, 1.600, 2.000, 2.400, 2.800, 3.200, 3.600, 4.000}");
            p("BC Values", "Array(double)",
              "{0, 0.000, 0.218, 0.426, 0.613, 0.771, 0.891, 0.969, 1.000, 0.982, 0.918, 0.809}");
          } // Time Dependent DBC on NS ns_4 for DOF Z
          break;
        case 3:
          // Pull apart.
          p("DBC on NS ns_4 for DOF Z", 0.0);
          p("DBC on NS ns_4 for DOF Y", 0.0);
          { pl a("Time Dependent DBC on NS ns_4 for DOF X");
            p("Time Values", "Array(double)", "{0.0, 0.2, 40.0}");
            p("BC Values", "Array(double)", "{0.0, 0.0, 20.0}");
          } // Time Dependent DBC on NS ns_4 for DOF X
          break;
        case 4:
          // Squeeze.
          p("DBC on NS ns_4 for DOF Z", 0.0);
          p("DBC on NS ns_4 for DOF Y", 0.0);
          { pl a("Time Dependent DBC on NS ns_4 for DOF X");
            p("Time Values", "Array(double)", "{0.0, 0.2, 2.2}");
            p("BC Values", "Array(double)", "{0.0, 0.0, -1.0}");
          } // Time Dependent DBC on NS ns_4 for DOF X
          break;
        }
      } // Dirichlet BCs
      { pl a("Parameters");
        p("Number", 1);
        p("Parameter 0", "Time");
      } // Parameters
      { pl a("Response Functions");
        p("Number", 1);
        p("Response 0", "Solution Average");
      } // Response Functions
      if (adapt) {
        pl a("Adaptation");
        if ( ! uniform) {
          p("Method", "RPI SPR Size");
          p("Error Bound", (quad ? 0.05 : 0.1));
        } else {
          p("Method", "RPI Unif Size");
          p("Target Element Size", target_element_size);
        }
        p("Remesh Strategy", "Continuous");
        p("Max Number of Mesh Adapt Iterations", 10);
        p("State Variable", "Cauchy_Stress");
        p("Transfer IP Data", true);
        p("Reference Configuration: Update", rcu);
        p("Reference Configuration: Transform", ! notransform);
        p("Reference Configuration: Project", project);
      } // Adaptation
    } // Problem
    { pl a("Discretization");
      p("Method", "PUMI");
      p("Workset Size", 500000);
      p("Mesh Model Input File Name", std::string("../../meshes/cube/cube") + (quad ? "-quad" : "") + ".dmg");
      p("PUMI Input File Name",
        std::string("../../meshes/cube/cube") + (quad ? "-quad" : "") + (parallel ? "" : "-serial") + ".smb");
      p("PUMI Output File Name", outfn);
      p("PUMI Write Interval", write_interval);
      p("Element Block Associations", "TwoDArray(string)", "2x1:{95, eb_1}");
      p("Node Set Associations", "TwoDArray(string)", "2x4:{85, 83, 43, 81, ns_1, ns_2, ns_3, ns_4}");
      p("2nd Order Mesh", quad);
      p("Cubature Degree", 2);
      // Refine the mesh on input so the worksets are large.
      if (resize0) {
        p("Max Number of Mesh Adapt Iterations", 10);
        p("Resize Input Mesh Element Size", (refine0 ? 1 : 0.5)*target_element_size);
      }
    } // Discretization
    { pl a("Regression Results");
      p("Number of Comparisons", 1);
      p("Test Values", "Array(double)", "{4.35233052684e-05}");
      p("Relative Tolerance", 0.01);
    } // Regression Results
    { pl a("Piro");
      { pl a("LOCA");
        { pl a("Bifurcation");
        } // Bifurcation
        { pl a("Constraints");
        } // Constraints
        { pl a("Predictor");
          p("Method", "Constant");
        } // Predictor
        { pl a("Stepper");
          p("Continuation Method", "Natural");
          p("Initial Value", 0.0);
          p("Continuation Parameter", "Time");
          p("Max Steps", max_steps);
          p("Max Value", 50.0);
          p("Min Value", 0.0);
          p("Compute Eigenvalues", false);
          p("Skip Parameter Derivative", true);
          { pl a("Eigensolver");
            p("Method", "Anasazi");
            p("Operator", "Jacobian Inverse");
            p("Num Eigenvalues", 0);
          } // Eigensolver
        } // Stepper
        { pl a("Step Size");
          p("Method", "Constant");
          p("Initial Step Size", step_size);
        } // Step Size
      } // LOCA
      { pl a("NOX");
        { pl a("Direction");
          p("Method", "Newton");
          { pl a("Newton");
            { pl a("Linear Solver");
              p("Tolerance", 1e-12); }
            p("Forcing Term Method", "Constant");
            p("Rescue Bad Newton Solve", true);
            { pl a("Stratimikos Linear Solver");
              { pl a("NOX Stratimikos Options");
              } // NOX Stratimikos Options
              { pl a("Stratimikos");
                p("Linear Solver Type", "Belos");
                { pl a("Linear Solver Types");
                  { pl a("AztecOO");
                    { pl a("VerboseObject");
                      p("Verbosity Level", "none");
                    } // VerboseObject
                    { pl a("Forward Solve");
                      { pl a("AztecOO Settings");
                        p("Aztec Solver", "GMRES");
                        p("Convergence Test", "r0");
                        p("Size of Krylov Subspace", 200);
                        p("Output Frequency", 10);
                      } // AztecOO Settings
                      p("Max Iterations", 200);
                      p("Tolerance", 1e-10);
                    } // Forward Solve
                  } // AztecOO
                  { pl a("Belos");
                    { pl a("VerboseObject");
                      p("Verbosity Level", "none");
                    } // VerboseObject
                    p("Solver Type", "Block GMRES");
                    { pl a("Solver Types");
                      { pl a("Block GMRES");
                        p("Convergence Tolerance", 1e-10);
                        p("Output Frequency", 10);
                        p("Output Style", 1);
                        p("Verbosity", 33);
                        p("Maximum Iterations", 200);
                        p("Block Size", 1);
                        p("Num Blocks", 200);
                        p("Flexible Gmres", false);
                      } // Block GMRES
                    } // Solver Types
                  } // Belos
                } // Linear Solver Types
                p("Preconditioner Type", "Ifpack2");
                { pl a("Preconditioner Types");
                  { pl a("Ifpack2");
                    p("Overlap", 2);
                    p("Prec Type", "ILUT");
                    { pl a("Ifpack2 Settings");
                      p("fact: drop tolerance", 0.0);
                      p("fact: ilut level-of-fill", 1.0);
                      p("fact: level-of-fill", 1);
                    } // Ifpack2 Settings
                  } // Ifpack2
                } // Preconditioner Types
              } // Stratimikos
            } // Stratimikos Linear Solver
          } // Newton
        } // Direction
        { pl a("Line Search");
          { pl a("Full Step");
            p("Full Step", 1.0);
          } // Full Step
          p("Method", "Full Step");
        } // Line Search
        p("Nonlinear Solver", "Line Search Based");
        { pl a("Printing");
          p("Output Precision", 3);
          p("Output Processor", 0);
          { pl a("Output Information");
            p("Error", true);
            p("Warning", true);
            p("Outer Iteration", true);
            p("Parameters", false);
            p("Details", false);
            p("Linear Solver Details", false);
            p("Stepper Iteration", true);
            p("Stepper Details", true);
            p("Stepper Parameters", false);
          } // Output Information
        } // Printing
        { pl a("Solver Options");
          p("Status Test Check Type", "Complete");
        } // Solver Options
        { pl a("Status Tests");
          int ntests = 0;
          const bool use_and = false;//true;
          p("Test Type", "Combo");
          p("Combo Type", use_and ? "AND" : "OR");
          { pl a(strint("Test ", ntests++));
            p("Test Type", "NormF");
            p("Norm Type", "Two Norm");
            p("Scale Type", "Scaled");
            p("Tolerance", 1e-12);
          }
          if ( ! use_and) { pl a(strint("Test ", ntests++));
            p("Test Type", "MaxIters");
            p("Maximum Iterations", 15);
          }
          { pl a(strint("Test ", ntests++));
            p("Test Type", "NormF");
            p("Scale Type", "Unscaled");
            p("Tolerance", 1e-10);
          }
          if ( ! use_and) { pl a(strint("Test ", ntests++));
            p("Test Type", "FiniteValue");
          }
          if (use_and) { pl a(strint("Test ", ntests++));
            p("Test Type", "NormUpdate");
            p("Tolerance", 1e-8);
          }
          p("Number of Tests", ntests);
        } // Status Tests
      } // NOX
    } // Piro
  }
}
