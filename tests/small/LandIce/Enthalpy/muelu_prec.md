Albany input
============

A variety of AMG options have been added and tested with the following prospect problems:

humboldt-3-20km (32 procs)  humboldt-1-10km (32 procs)  green-3-20km (32 procs)   green-1-10km (256 procs)  thwaites-1-10km (384 procs)

using 10 layers.

Main MueLu Grid Transfer options
================================
All AMG options use semi-coarsening to generate the 2nd AMG level in hierarchy. This 2nd level is a 1 layer version of the problem obtained by using any of the following interpolation operators

   P1semi:  symmetric-style (or traditional) operator-dependent semi-coarsening interpolation from 2nd level to 1st level.
            This corresponds to the old (from several years ago) method used for non-enthalpy ice sheet formulations.
   P1const: piecewise constant interpolation from 2nd level to 1st level
   P1lin:   linear interpolation from 2nd level to 1st level

and any of the following restriction operators

    R1trans: transposed interpolation (any of the above options) from 1st level to 2nd level. So, P1semiR1trans is traditional method.
    R1const: piecewise constant interpolation from 1st level to 2nd level
    R1semi:  a non-symmetric version of P1semi aimed at producing restriction from level 2 to level 1
    R1lin:   linear restriction from 1st level to 2nd level

Note: there is some naming redundancy. For example, P1linRtrans is the same as P1linRlin.

Additional AMG levels are obtained by then applying more standard AMG coarsening algorithms. These coarser level interpolation options include

    P2const: piececonstant interpolation
    P2sa:    standard smoothed aggregation
    P2pg:    Petrov-Galerkin smoothed aggregation, a non-symmetric version of smoothed aggregation

For P2const and P2sa, coarser level restriction is always the transpose of interpolation. For P2pg, a restriction operator oriented toward non-symmetric systems is generated. All P2* options apply the same grid transfer algorithms to generate all coarser level operators (i.e., only at the finest level is a different semi-coarsening algorithm applied to generate grid transfers between the 1st and 2nd levels).

MueLu Smoother options
======================
3 main smoothers can be run on the finest level
     line:        pre and post line relaxation 
     plane:       pre and post plane relaxation
     plane+line : pre plane smoothing  and post line smoothing

On coarser levels, we always perform Gauss-Seidel smoothing.

All algorithms can be used in many different possible combinations.


Quick Observation on Results
============================
1) None of the enthapy problems seemed super difficult for AMG, though the solver for green-1-10km does not seem scalable.

2) line relaxation did much much better than plane relaxation. This is different from what I saw when starting out on the enthalpy formulation.  Not sure what has changed but there was a time that plane relaxation definitely was needed. Plane relaxation was so bad, I omit it from the results shown below.

3) Even though all linear solvers are solved with a fairly tight tolerance, there is a noticeable variation in nonlinear its, which somewhat clouds an evaluation of linear solvers. It sort of seems that some linear solvers give a better answer than others from a nonlinear perspective.

4) If I had to just pick one set of grid transfer choices over all problems, I'd probably go with P1semiR1transP2const. This uses the traditional algorithm at the finest levels and piecewise constant interpolation for the coarser levels. There are definitely other combinations that are either competitive with this or slightly beat it, but this seems like a good overall choice. Not sure if things would be different for Antarctica, where we would have ice shelves.


High Level Results
==================
MainOptions gives the grid transfers algorithms used at the finest levels (P1 and R1) and at coarser levels (P2). ItersPerSolves give the number of linear its per non-linear solve iteration to reduce residual by 1.0e-7. AlbanyTotalTime gives the total Albany time. Output sorted based on AlbanyTotalTime.

1) humboldt-3-20km

full Output file on Cori

     /global/cscratch1/sd/rstumin/prospect/Jan2022/ali-perf-tests/perf_tests/humboldt-3-20km/enthOutFri_Jun_24_07:29:45_PDT_2022

Relevant output can be obtained via

      /global/homes/r/rstumin/bin/skimOutput enthOutFri_Jun_24_07:29:45_PDT_2022 | /global/homes/r/rstumin/bin/recordSort

I can recreate complete yamls by running the script /global/homes/r/rstumin/bin/recordSort/ttRun

                      MainOptions         ItsPerSolve     AlbanyTotalTime
                   P1linR1constP2sa     (14,18,22,22,21)         3.5
                   P1linR1transP2pg     (16,20,22,23,21)         3.5
                   P1linR1transP2const  (16,20,22,23,21)         3.5
                   P1constR1transP2sa   (15,19,21,22,20)         3.6
                   P1semiR1transP2sa    (13,18,21,22,20)         3.7
                   P1linR1transP2sa     (16,20,22,23,21)         3.7
                   P1constR1transP2const(15,19,21,22,20)         3.8
                   P1linR1constP2const  (14,19,22,23,21)         3.8
                   P1linR1constP2pg     (14,19,22,23,21)         3.9
                   P1semiR1linP2sa      (15,21,22,23,22)         4.0
                   P1semiR1constP2const (14,19,22,23,21)         4.3
                   P1semiR1transP2pg    (13,18,21,22,20)         4.6
                   P1semiR1linP2pg      (14,22,23,24,21)         4.8
                   P1constR1transP2pg   (15,19,21,22,20)         5.9
                   P1semiR1transP2const (13,18,21,22,20)         6.1
                   P1semiR1constP2sa    (14,19,22,23,21)         6.3
                   P1semiR1linP2const   (14,22,23,23,21)         6.5
                   P1semiR1constP2pg    (14,18,22,23,21)         9.8
                   P1semiR1semiP2const  (~75 linIts,manyNLits)  71.8
                   P1semiR1semiP2sa     (>75 linIts,manyNLits)  90.8
                   P1semiR1semiP2pg     (>75 linIts,manyNLits) 100.5

Observations: Given the timing variations, all the non-P1semiR1semi* runs are all pretty much fine. One can see that the slow P1semiR1constP2pga (9.8 seconds) pretty much take the same number of iterations as the fasterst P1linR1constP2sa. The setup time might be a bit slower for this slow option, but I wouldn't conclude too much from these runs other than the fact that non-symmetric semi-coarsening (R1semi) performs poorly.  Upon examination, it seems like something bad happens near the boundaries. Not a code bug, but a somewhat strange restriction operator near the boundaries.  It's also worth noting that relatively simple algorithms (e.g., piecewise constants for all grid transfers such as P1constR1transP2const) are competitive with fancier algorithms.

2) humboldt-1-10km

full Output file on Cori

     /global/cscratch1/sd/rstumin/prospect/Jan2022/ali-perf-tests/perf_tests/humboldt-1-10km/enthOutFri_Jun_24_07:37:30_PDT_2022

Relevant output can be obtained via

     /global/homes/r/rstumin/bin/skimOutput enthOutFri_Jun_24_07:37:30_PDT_2022 | /global/homes/r/rstumin/bin/recordSort

                      MainOptions         ItsPerSolve                                                               AlbanyTotalTime
                   P1linR1constP2sa     (14,15,17,19,19,19,18,17,18,17,18,18,17,18,18,18)                                   43.8
                   P1semiR1transP2const (14,17,17,18,18,18,20,17,17,17,18,17,17,15,17,16,16,15)                             47.7
                   P1semiR1constP2sa    (14,16,18,19,20,19,18,18,18,18,18,18,17,16,16,16,16)                                48.3
                   P1semiR1transP2sa    (13,17,17,18,18,17,17,17,17,16,17,17,17,17,17,17,17,16)                             49.8
                   P1semiR1transP2pg    (14,17,17,18,18,18,19,17,17,17,17,17,17,15,16,16,16,15)                             51.5
                   P1semiR1constP2const (14,16,18,19,20,19,18,18,18,18,18,18,18,18,18,18,18,14,15)                          51.6
                   P1semiR1linP2const   (14,18,18,20,20,19,18,18,18,18,17,17,17,18,18,17,17,17,16,16)                       53.0
                   P1constR1transP2sa   (14,14,17,19,19,19,20,19,19,19,19,19,19,19,18,18,17,14,12,11)                       53.6
                   P1semiR1linP2sa      (14,18,18,19,20,19,18,18,18,18,17,17,17,18,18,18,17,17,17,17)                       53.7
                   P1linR1constP2pg     (14,15,17,19,19,20,18,18,18,18,18,18,18,17,17,18,17,17,16,16,16)                    56.3
                   P1semiR1linP2pg      (14,18,18,20,20,19,19,18,18,18,18,17,18,17,18,18,17,13,11,11)                       57.2
                   P1linR1constP2const  (14,15,17,19,19,20,18,18,17,18,18,18,18,17,17,18,17,17,16,16,16,15)                 58.4
                   P1constR1transP2const(14,14,17,19,19,19,20,19,19,19,19,18,19,19,19,19,19,19,19,19,19,19)                 59.4 
                   P1semiR1constP2pg    (14,16,18,19,20,19,18,18,18,18,17,18,18,18,17,17,16,15,15,14,13)                    59.9
                   P1constR1transP2pg   (14,14,17,19,19,19,20,19,19,19,19,18,19,19,19,19,19,19,19,19,19,19)                 62.7
                   P1linR1transP2const  (24,20,23,24,25,25,24,25,23,25,25,24,25,24,24,24,24,24,25,23,23,24,23,24)           69.8 
                   P1linR1transP2pg     (24,20,23,24,25,25,24,25,23,25,25,24,25,24,24,24,24,24,25,23,23,24,23,24)           70.8
                   P1semiR1semiP2sa     (32,39,67,52,36,48,47,55,58,50,52,53,53,49,49,48,48,48,45)                          78.1
                   P1linR1transP2sa     (24,20,23,24,25,25,24,25,23,25,25,24,25,24,24,24,24,24,25,23,23,23,23,21,22,19,20)  81.4     
                   P1semiR1semiP2const  (28,55,65,52,51,59,61,60,62,61,60,60,60,60,60,58,60,59,60,59,59,58,60)             101.7
                   P1semiR1semiP2pg     (26,47,62,50,40,39,38,42,50,47,43,39,52,52,48,53,56,57,57,51,51,50,52,52,54,52,54, 234.1
                                         55,54,55,54,54,54,54,52,52,53,53,49,49,56,54,53,52,52,53,53,54,53,56)

Observations: Not sure why such variation in # of nonlinear steps, even though all linear solvers reduce residual by 1e-7. The maximum allowed linear iterations per linear solve was set to 200 for all runs.  Perhaps a newer version of Albany would perform differently? The varying # of nonlinear steps affects the assessment of linear solvers. It is not clear whether some linear solvers give a "better" solution than others leading to fewer nonlinear iterations?  The number of linear iterations per nonlinear solve is about the same as for humbold-3-20km for many of the options, which is nice. One doesn't really see a jump in linear its per solve until the last 6 runs (P1linR1transP2* , and the *R1semi* options). So it seems that linear interpolation for both P1 and R1 is not great, though it was fine when only used for P1 or R1 (but not both). Sometimes linear interpolation/restriction can be a bit problematic for non-symmetric problems in terms of the stability of the coarse grid operators similar to standard finite element discretization of nonsymmetric operators. So, I don't consider this linear interpolation/restriction observation to be too unusual. There is no clear winner among the coarser level grid transfer options (P2sa or P2const or P2pg). That is, these P2 options don't seem to have much impact on the results.


3) green-3-20km
      
full Output file on Cori

     /global/cscratch1/sd/rstumin/prospect/Jan2022/ali-perf-tests/perf_tests/green-3-20km/enthOutFri_Jun_24_08:40:25_PDT_2022

Relevant output can be obtained via

     /global/homes/r/rstumin/bin/skimOutput enthOutFri_Jun_24_08:40:25_PDT_2022 | /global/homes/r/rstumin/bin/recordSort

                      MainOptions         ItsPerSolve                                                               AlbanyTotalTime

                   P1semiR1linP2sa      (11,10,12,12,11,12,12,12,11,10,10,11,11,11,12,12,12,12,11,11,11)                171.9
                   P1semiR1linP2const   (11,10,12,13,11,12,12,12,11,10,13,13,15,12,10,12,11,11,11,11,10,11)             175.8
                   P1semiR1transP2sa    (11,10,12,12,11,12,12,12,11,10,11,11,11,11,11,11,12,12,12,12,12,10)             176.1
                   P1constR1transP2pg   (11,10,12,12,11,11,12,11,11,11,13,14,14,13,11,13,12,12,12,12,11,11)             177.9
                   P1semiR1transP2pg    (11,10,12,13,11,12,12,12,11,10,11,11,11,11,11,10,12,12,12,12,12,12)             178.2
                   P1semiR1constP2pg    (11,10,12,12,11,11,11,11,10,10,10,11,10,10,11,11,12,12,11,11,11,11)             179.3
                   P1linR1constP2const  (12,12,13,14,12,12,12,12,11,11,12,13,12,13,13,12,13,13,13,13,13,12)             179.6
                   P1constR1transP2sa   (11,10,12,12,11,11,12,11,11,11,13,14,14,13,11,12,12,12,12,12,11,11)             180.0
                   P1constR1transP2const(11,10,12,12,11,11,12,11,11,11,13,14,14,13,11,13,12,12,12,12,11,11)             180.1
                   P1linR1constP2pg     (12,12,13,14,12,12,12,12,11,11,12,13,12,13,13,12,13,13,13,13,12,12)             180.6
                   P1semiR1linP2pg      (11,11,12,13,11,12,12,12,11,11,11,11,11,11,11,11,11,13,13,12,12,12,11)          186.6
                   P1semiR1constP2sa    (11,10,12,12,11,11,11,11,10,10,10,11,10,12,11,11,12,11,11,11,11,11,11,11)       193.5
                   P1semiR1constP2const (11,10,12,12,11,11,12,11,10,10,10,11,10,12,11,11,12,11,11,11,11,11,11,11)       194.0
                   P1semiR1transP2const (11,10,12,13,11,12,12,12,11,10,11,11,10,11,11,10,10,12,11,11,11,11,11,11)       194.2
                   P1linR1transP2const  (16,13,15,17,16,17,17,17,17,17,17,17,17,17,17,17,17,19,18,17,17,17,14)          201.2
                   P1linR1transP2pg     (16,13,15,17,16,17,17,17,17,17,17,17,17,17,17,17,17,19,18,17,17,17,14)          202.1
                   P1linR1transP2sa     (16,13,15,17,16,17,17,17,17,17,17,17,17,17,17,17,17,19,18,17,17,17,14)          203.0
                   P1linR1constP2sa     (12,12,13,14,12,12,12,12,11,11,11,11,12,11,12,12,11,13,13,13,12,12,12,12,12,12,11)218.5
                   P1semiR1semiP2const  ( bad )                                                                         457.9
                   P1semiR1semiP2pg     ( bad )                                                                         711.0
                   P1semiR1semiP2sa     ( bad )                                                                        1574.5


Observations: Still have variation in # of nonlinear steps. Nonsym semi-coarsening is again bad. Linear interp semicoarsening seems less good when used in conjunction with linear restriction (P1linR1trans*) but linear restriction used with a different interpolation algorithm seems fine. Again, no clear winner among the P2* algorithms.

4) green-1-10km
      
full Output file on Cori

     /global/cscratch1/sd/rstumin/prospect/Jan2022/ali-perf-tests/perf_tests/green-1-10km/enthOutFri_Jun_24_10:51:49_PDT_2022

Relevant output can be obtained via

     /global/homes/r/rstumin/bin/skimOutput enthOutFri_Jun_24_10:51:49_PDT_2022 | /global/homes/r/rstumin/bin/recordSort

                      MainOptions                   ItsPerSolve                      AlbanyTotalTime
                   P1semiR1constP2const  (25,33,32,33,31,31,30,29,29)                    109.0
                   P1semiR1transP2const  (26,34,34,33,32,32,31,31,31)                    109.4
                   P1semiR1linP2const    (26,35,37,33,32,31,33,29,30)                    109.6
                   P1linR1constP2const   (27,34,34,33,33,32,32,31,31)                    110.2
                   P1linR1constP2pg      (26,33,34,33,32,32,31,31,31)                    111.6
                   P1constR1transP2pg    (25,34,34,34,34,33,33,32,32)                    111.7
                   P1constR1transP2const (25,34,34,34,34,33,33,32,32)                    112.0
                   P1semiR1linP2pg       (27,38,38,45,32,31,32,29,30)                    113.2
                   P1semiR1constP2pg     (26,35,35,34,36,35,34,33,33)                    113.9
                   P1semiR1transP2pg     (29,37,36,36,35,35,34,34,34)                    114.4
                   P1linR1transP2const   (47,52,51,51,51,50,50,49,49)                    136.3
                   P1linR1transP2sa      (47,52,51,51,51,50,50,49,49)                    136.5
                   P1linR1transP2pg      (47,52,51,51,51,50,50,49,49)                    139.1
                   P1semiR1transP2sa     (43,55,57,55,56,54,54,53,53)                    143.4
                   P1constR1transP2sa    (70,79,79,76,76,84,73,76,72)                    177.6
                   P1semiR1linP2sa       (127,144,38,48,36,29,45,64,57,185,113,29,
                                          172,188,179,174,177,53,146,187,179)            543.9
                   P1linR1constP2sa      (194,190,192,188,188,185,183,182,177,176,
                                          175,175,174,174,174,174)                       606.7
                   P1semiR1constP2sa     (182,76,45,26,92,61,170,37,100,187,190,181,
                                          186,180,183,178,176,176,176,174,177,153,175)   719.4
                   P1semiR1semiP2const   (bad)
                   P1semiR1semiP2sa      (bad)
                   P1semiR1semiP2pg      (bad) Failed (Throw, Lapack band factorization failed)

Observations: lin iters per solve grows when the mesh is refined, which is not nice.  Non-symmetric semi-coarsen restriction is again bad. The use of linear interp with linear restriction (P1linR1trans) is again not the best. What stands out most, is that P2sa (using SA on coarser levels) is almost always inferior to the other options. For example, P1semiR1constP2const is considerably better than P1semiR1constP2sa). Similar to linear interpolation, SA can have problems on highly non-symmetric systems and so it seems that P2const and P2pg pay off here in a significant way.

5) thwaites-1-10km

full Output file on Cori

     /global/cscratch1/sd/rstumin/prospect/Jan2022/ali-perf-tests/perf_tests/thwaites-1-10km/enthOutFri_Jun_24_15:10:31_PDT_2022

Relevant output can be obtained via

     /global/homes/r/rstumin/bin/skimOutput enthOutFri_Jun_24_15:10:31_PDT_2022 | /global/homes/r/rstumin/bin/recordSort



                   P1semiR1linP2const    (48,45,19,20,21,19,14,10,10,9,9,10,10)     23.5
                   P1constR1transP2pg    (42,38,22,23,23,20,15,9,9,9,9,10,9)        23.8
                   P1constR1transP2sa    (42,38,22,22,23,20,15,8,9,9,9,10,9)        24.5
                   P1linR1constP2const   (45,38,18,19,23,19,14,10,13,12,10,12,11)   24.7
                   P1linR1constP2pg      (45,38,18,18,22,19,14,10,12,11,10,12,11)   24.8
                   P1semiR1transP2sa     (42,37,26,26,27,24,14,10,9,8,8,9,9)        25.0
                   P1constR1transP2const (42,38,22,23,24,20,15,9,9,9,9,10,9)        25.2
                   P1semiR1transP2const  (42,37,27,27,28,25,14,11,9,8,8,9,9)        25.2
                   P1linR1constP2sa      (46,39,20,21,25,23,17,11,13,12,10,12,12)   25.5
                   P1semiR1linP2pg       (48,44,19,20,21,19,14,10,10,9,9,10,10)     25.6
                   P1semiR1transP2pg     (42,37,27,27,28,25,14,11,9,8,8,9,9)        26.2
                   P1semiR1constP2pg     (52,47,22,23,25,25,19,12,12,12,11,12,12)   27.1
                   P1semiR1constP2const  (51,46,21,21,23,23,17,11,11,10,10,11,11)   28.6
                   P1semiR1constP2sa     (56,52,30,30,32,28,26,17,15,14,13,14,14)   28.8
                   P1linR1transP2const   (62,61,51,51,52,48,22,23,15,14,13,14,14)   32.0
                   P1linR1transP2pg      (62,61,51,51,52,48,22,23,15,14,13,14,14)   32.8
                   P1linR1transP2sa      (62,61,51,51,52,48,22,23,15,14,13,14,14)   32.9
                   P1semiR1linP2sa       (62,64,52,51,59,56,47,23,13,14,13,15,15)   35.3
                   P1semiR1semiP2const   (bad)                                     480.3
                   P1semiR1semiP2sa      (bad)                                     485.5
                   P1semiR1semiP2pg      (bad)                                     Failed (Throw, Lapack band factorization failed)


Observations: Non-symmetric semi-coarsen restriction is bad. Most other options seem fine, though the P1linR1trans* options are clearly not the best. It seems like the P1semiR1const* options are also not the best. In this case, it is hard to draw solid conclusions about the P2* options. There are also some curiosities in the output. For example, P1semiR1linP2const is noticeably better than P1semiR1linP2sa, but there are other cases where P2sa was fine. Perhaps the use of R1lin in conjunction with P2sa lead to some stability problems on coarser levels? It is hard to say and I'm just speculating. Overall, I'd say that it is safer to go with something like P2const as it is never far from the best P2* option.

