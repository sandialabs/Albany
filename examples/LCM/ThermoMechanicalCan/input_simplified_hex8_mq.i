$ Aprepro (Revision: 2.30) Thu Aug 15 14:46:40 2013
  Begin sierra adagio

# Restart is automatic

Title Mech-Only with Constant Temperature and Linear Pressure in FIC

# Dempsey/Dodd/Black - Upright Can #6 Simulation
# units in the file are SI (m, N, s, kg, Pa, W, J, K)
   
	define direction pos_x_axis with vector 1.0 0.0 0.0
	define direction pos_y_axis with vector 0.0 1.0 0.0
	define direction pos_z_axis with vector 0.0 0.0 1.0	

############################################################
#                                                          #
#        ADAGIO materials models properties and functions  #
#                                                          #
############################################################
   
  begin property specification for material 304-tube

    density         = 7920 #Kg/m^3 7.32e-4 lb-s^2/in^4
    thermal engineering strain function = 304-tube_thermal_strain

    begin parameters for model ml_ep_fail
      youngs modulus     = {193.1E09*0.882}  #Pa 28E06 psi
      poissons ratio     = {0.28*1.143}
      yield stress       = {241.8E06*0.343}  #Pa 35072.0 psi
      beta               = 1.0
      critical tearing parameter  =  1.e4
      critical crack opening strain =  0.17
      hardening function = hard_973
    end parameters for model ml_ep_fail

  end property specification for material 304-tube

  begin property specification for material 304-lid

    density         = 7920 #Kg/m^3 7.32e-4 lb-s^2/in^4
    thermal engineering strain function = 304-lid_thermal_strain

    begin parameters for model ml_ep_fail
      youngs modulus     = {193.1E09*0.571291}  #Pa 28E06 psi
      poissons ratio     = {0.28*1.143}
      yield stress       = {247.E06*0.356339}  #Pa 35800.0 psi
      beta               = 1.0
      critical tearing parameter  =  1.e4
      critical crack opening strain =  0.17
      hardening function = hardl_973
    end parameters for model ml_ep_fail

  end property specification for material 304-lid
  
#############################################################
#   below are the functions for the material properties
#   youngs   - normalized Young's modulus as a function of temperature
#   poissons - normalized Poisson's ratio as a function of temperature
#   yield    - normalized yield stress as a function of temperature
#############################################################
   
  begin definition for function hard_973
    type is piecewise linear
    begin values
      includefile materials/700C_18NA_hardening_function.dat
    end values
  end definition for function hard_973

  begin definition for function hardl_973
    type is piecewise linear
    begin values
      includefile materials/700C_19NAL_hardening_function.dat
    end values
  end definition for function hardl_973

# Needs to be updated with temperature dependent CTE
#
   Begin Definition for Function 304-tube_thermal_strain
      Type = Piecewise Linear
      #Abscissa = Temperature # K
      #Ordinate = Strain # 1/K
      #Y Scale = 17.3e-6 # K^-1 (this is the CTE)
      Begin Values # alpha = 17.3e-6
         0      -5.069e-3
	 293	0.0
         1293	17.3e-3
      End
   End

# Repeated the same function for lid as for tube above
# Needs to be updated with temperature dependent CTE
#
   Begin Definition for Function 304-lid_thermal_strain
      Type = Piecewise Linear
      #Abscissa = Temperature # K
      #Ordinate = Strain # 1/K
      #Y Scale = 17.3e-6 # K^-1 (this is the CTE)
      Begin Values # alpha = 17.3e-6
         0      -5.069e-3
	 293	0.0
         1293	17.3e-3
      End
   End

  begin function elementTri
     Expression Variable: von = element von_mises
     Expression Variable: mean = element hydrostatic_stress
     type = analytic
     evaluate expression = "mean/von"
  end

    Begin Definition for Function internal_pressure
       Type = Piecewise Linear
       Begin Values  #Time - Gauge Pressure (Pa)
	0   0.0	   
        985.2   7152483.32
	3600    26135749.04	 
        End
     End Definition for Function internal_pressure

###############################################################
#                                                             #
#     ADAGIO mesh                                             #
#                                                             #
###############################################################   
   
  begin solid section solid_1
    strain incrementation = strongly_objective
$ KHP: Add best hourglass formulation
    hourglass formulation = total
  end solid section solid_1

   Begin Finite Element Model short-can-mechanical     
      Database Name = mesh/mech90-mesh4-base3-SI.g
      Database Type = ExodusII

      Begin Parameters for block block_1 block_2 block_4
         Material 304-tube
         Solid Mechanics Use Model ml_ep_fail
         section = solid_1
      End

      Begin Parameters for block block_3 block_5
         Material 304-lid
         Solid Mechanics Use Model ml_ep_fail
         section = solid_1
      End

   End Finite Element Model short-can-mechanical

################################################################
#     Time Control                                             #
################################################################  
   
   
 	Begin Adagio procedure myProcedure
	
        begin solution control description      
    
	Use system Main
	Begin system main
          Simulation Start Time = 0
          Simulation Termination Time = 3600
   	  simulation Max Global Iterations = 10000
	
	  Begin transient solution_block_all 
	   advance adagio
	  End transient solution_block_all

        End system main
    
	  Begin parameters for Transient solution_block_all
	    Start time = 0.0
	    Termination time = 3600
	    begin parameters for adagio region adagio
	        time increment = 1.0 # previously used 10.0
	    End parameters for adagio region adagio
	  End
      
        end solution control description     

    $=========================================================
    $  End ofsolver control parameters
    $=========================================================

###############################################
#                                             #
#    ADAGIO solver and results                #
#                                             #
###############################################


      Begin Adagio Region adagio 
      
         Use Finite Element Model short-can-mechanical
      
         # nonlinear solver parameters
         Begin Solver
            
#            Begin Loadstep Predictor
######               Type = Secant
#               type = scale_factor
#               scale factor = 1.0
#            End Loadstep Predictor

         begin control failure fred
            maximum iterations = 100
            level = 1
         end control failure fred
   
            # this defines the nonlinear CG method using FETI
            Begin CG
            
#               Reference = Internal
               Target Relative Residual = 1e-6
               acceptable relative residual = 1e-4
               acceptable residual = 1.0
               maximum iterations       = 20
#               cd       = 3
               # Acceptable Relative Residual = 1 # take this out for real runs
#               Iteration Reset = 500
               Line Search Secant
   
               # use a full tangent preconditioner on the equations for
               # each step of the nonlinear CG method.  Note: this is much
               # faster than the default "Preconditioner = Elastic" choice
               line search secant 1e-3
               Begin Full Tangent Preconditioner	       
	       iteration update = 10
                  Linear Solver = feti
$ KHP: This line command turns off an expensive diagnostic check
$ KHP: and thus is mostly for improving speed.
                  conditioning  = no_check
                  # Constraint Enforcement = Solver
                  # Minimum Convergence Rate = 1e-6 # be more leniant than the default of 1e-4
                  # Adaptive Strategy = Update # update preconditioner instead of switching to default scheme
               End
            End CG
   
         End Solver
	 
#      begin adaptive time stepping
#        cutback factor = 0.5
#        growth factor  = 2.0 # previously 1.5
# let it be default Target/5:  previously iteration window  = 100
#	maximum failure cutbacks = 15
#	maximum multiplier = 100
#	minimum multiplier = 1.e-8 #1.e-12 previously
$ KHP: Set target iterations equal to the maximum iterations in the solver block.
#	target iterations = 20 #400 previously
#      end
  
#
#  Calculate cavity volume - with/without symmetry (by Nathan Crane)
#
          begin user output
            SURFACE = surface_1
            SURFACE SUBROUTINE = aupst_cavity_volume
            #subroutine string parameter: outputVarName = foam_vol
            subroutine real   parameter: cavitySymmetryPoint_x = 0
            subroutine real   parameter: cavitySymmetryPoint_y = 0
            subroutine real   parameter: cavitySymmetryPoint_z = 0
            subroutine real   parameter: volumeScale           = 4.0
	    subroutine string parameter: outputVarName = free_vol
            compute at every step
          end

      begin user output
           include all blocks
           compute element triaxiality as function elementTri
      end
	  
          begin user variable free_vol #foam_vol  ABD -- this it the volume generated due to expansion of the can
            TYPE = GLOBAL REAL LENGTH = 1
            GLOBAL OPERATOR = MAX
            INITIAL VALUE = 0.0  
          end user variable free_vol

#---- Restart for adagio
 
      begin restart data restart_1
        database type = exodusII
        database Name = restarts/adagio.rsout
        At time 0, Increment = 10.0
        overlay count = 99999 # saves only the last one
       end      	  	    
                
      # results output for Adagio
      begin user output
        block = block_1
        compute global block_1_max_tear as max of element tearing_parameter
        compute global block_1_max_eqps as max of element eqps
        compute global block_1_avg_tear as average of element tearing_parameter
        compute global block_1_avg_eqps as average of element eqps 
        compute global block_1_avg_press as average of face pressure
      end user output

      begin user output
        block = block_2
        compute global block_2_max_tear as max of element tearing_parameter
        compute global block_2_max_eqps as max of element eqps
        compute global block_2_avg_tear as average of element tearing_parameter
        compute global block_2_avg_eqps as average of element eqps 
        compute global block_2_avg_press as average of face pressure
      end user output

       begin heartbeat output weld_1
        stream name = weld_1.txt
        timestamp format = ''
        variable = global time
        variable = global block_1_max_tear
        variable = global block_1_max_eqps
        variable = global block_1_avg_tear
        variable = global block_1_avg_eqps
        variable = global block_1_avg_press
        at time 0.0, increment = 1
        labels = off
        legend = on
        precision = 7
       end heartbeat output weld_1

       begin heartbeat output weld_2
        stream name = weld_2.txt
        timestamp format = ''
        variable = global time
        variable = global block_2_max_tear
        variable = global block_2_max_eqps
        variable = global block_2_avg_tear
        variable = global block_2_avg_eqps
        variable = global block_2_avg_press
        at time 0.0, increment = 1
        labels = off
        legend = on
        precision = 7
       end heartbeat output weld_2

# results output for Adagio
         Begin Results Output adagio_results
            Database Name = results/mechanical.e
            At time 0.0, Increment = 1.0
            At time 10.0, Increment =10
#            At time 500.0, Increment =1
#            At time 1000.0, Increment =10
            Nodal Variables = displacement as disp
            Element Variables = effective_log_strain as log_strain
            Element Variables = von_mises as vm
	    Element Variables = max_principal_stress as smax
            Element Variables = stress as stress
            Element Variables = eqps as eqps
	    Element Variables = tearing_parameter as tear
            Element Variables = failure_flag
            element variables = rate_of_deformation as strain_rate
            element variables = triaxiality
            Global Variables = timestep
	    global variables = free_vol 
	    face Variables = pressure as pg
         End


###############################################################
#                                                             #
#     ADAGIO boundary conditions                              #
#                                                             #
###############################################################
	  
         Begin Fixed Displacement
            Node Set = nodelist_1
            Component = Z
         End  	 
         Begin Fixed Displacement
            Node Set = nodelist_2
            Component = X
         End 
	  	       
         # hold the bottom
#         Begin Fixed Displacement
#            Node Set = nodelist_3
#            Component = X Y Z
#         End 
#         # hold the bottom
#         Begin Fixed Displacement
#            Node Set = nodelist_4
#            Component = X Z
#         End 	 
         # hold the bottom
         Begin Fixed Displacement
            Node Set = nodelist_3
            Component = Y
         End 	 	 
	          
         # internal pressure
         Begin Pressure internal_pressure
            Surface = surface_1
            Function = internal_pressure
#           Scalar Source Variable = pg Index 0 Region aria # taken from the foam decomposition
         End

      End Adagio Region adagio

  end Adagio procedure myProcedure   

$ KHP: Adjust the feti solver settings to improve the speed
$ KHP: and hopefully not adversely affect the robustness.
#    param-string "preconditioner_solver" value "single_precision_sparse" -- Decided not to use 
#    residual norm tolerance = 1.e-3 and used instead 1.e-6
             begin feti equation solver feti
                 corner augmentation = edge
                 residual norm tolerance = 1.e-6
                 maximum iterations = 1000
#                 minimum iterations = 3
             end

end sierra adagio
