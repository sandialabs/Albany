def create_slip_systems(
    crystal_structure = 'fcc',
    ratio_c_a = 1.0,
    slip_families = None):

    slip_systems = []

    if crystal_structure == 'fcc':

        # system 1
        # { 1  1  1} <-1  1  0>
        direction = (-1.0, 1.0, 0.0)
        normal = (1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 2
        # { 1  1  1} < 0 -1  1>
        direction = (0.0, -1.0, 1.0)
        normal = (1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 3
        # { 1  1  1} < 1  0 -1>
        direction = (1.0, 0.0, -1.0)
        normal = (1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 4
        # {-1  1  1} <-1 -1  0>
        direction = (-1.0, -1.0, 0.0)
        normal = (-1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 5
        # {-1  1  1} < 1  0  1>
        direction = (1.0, 0.0, 1.0)
        normal = (-1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 6
        # {-1  1  1} < 0  1 -1>
        direction = (0.0, 1.0, -1.0)
        normal = (-1.0, 1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 7
        # {-1 -1  1} < 1 -1  0>
        direction = (1.0, -1.0, 0.0)
        normal = (-1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 8
        # {-1 -1  1} < 0  1  1>
        direction = (0.0, 1.0, 1.0)
        normal = (-1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 9
        # {-1 -1  1} <-1  0 -1>
        direction = (-1.0, 0.0, -1.0)
        normal = (-1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 10
        # { 1 -1  1} < 1  1  0>
        direction = (1.0, 1.0, 0.0)
        normal = (1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 11
        # { 1 -1  1} <-1  0  1>
        direction = (-1.0, 0.0, 1.0)
        normal = (1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

        # system 12
        # { 1 -1  1} < 0 -1 -1>
        direction = (0.0, -1.0, -1.0)
        normal = (1.0, -1.0, 1.0)
        slip_systems.append((direction, normal))

    elif crystal_structure == 'bcc':

        if '110' in slip_families:

            # system 1
            direction = (1.0, 1.0, -1.0)
            normal = (0.0, 1.0, 1.0)
            slip_systems.append((direction, normal))

            # system 2
            direction = (1.0, -1.0, 1.0)
            normal = (0.0, 1.0, 1.0)
            slip_systems.append((direction, normal))

            # system 3
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 4
            direction = (1.0, -1.0, 1.0)
            normal = (1.0, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 5
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 6
            direction = (1.0, 1.0, -1.0)
            normal = (1.0, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 7
            direction = (1.0, 1.0, 1.0)
            normal = (0.0, 1.0, -1.0)
            slip_systems.append((direction, normal))

            # system 8
            direction = (-1.0, 1.0, 1.0)
            normal = (0.0, 1.0, -1.0)
            slip_systems.append((direction, normal))

            # system 9
            direction = (1.0, 1.0, 1.0)
            normal = (-1.0, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 10
            direction = (1.0, 1.0, -1.0)
            normal = (-1.0, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 11
            direction = (1.0, 1.0, 1.0)
            normal = (1.0, 0.0, -1.0)
            slip_systems.append((direction, normal))

            # system 12
            direction = (1.0, -1.0, 1.0)
            normal = (1.0, 0.0, -1.0)
            slip_systems.append((direction, normal))

        if '112' in slip_families:

            # system 13
            direction = (1.0, -1.0, -1.0)
            normal = (2.0, 1.0, 1.0)
            slip_systems.append((direction, normal))

            # system 14
            direction = (-1.0, -1.0, 1.0)
            normal = (1.0, 1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 15
            direction = (-1.0, 1.0, -1.0)
            normal = (1.0, 2.0, 1.0)
            slip_systems.append((direction, normal))

            # system 16
            direction = (-1.0, -1.0, -1.0)
            normal = (-2.0, 1.0, 1.0)
            slip_systems.append((direction, normal))

            # system 17
            direction = (1.0, -1.0, 1.0)
            normal = (-1.0, 1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 18
            direction = (1.0, 1.0, -1.0)
            normal = (-1.0, 2.0, 1.0)
            slip_systems.append((direction, normal))

            # system 19
            direction = (1.0, -1.0, 1.0)
            normal = (2.0, 1.0, -1.0)
            slip_systems.append((direction, normal))

            # system 20
            direction = (-1.0, -1.0, -1.0)
            normal = (1.0, 1.0, -2.0)
            slip_systems.append((direction, normal))

            # system 21
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, 2.0, -1.0)
            slip_systems.append((direction, normal))

            # system 22
            direction = (1.0, 1.0, -1.0)
            normal = (2.0, -1.0, 1.0)
            slip_systems.append((direction, normal))

            # system 23
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, -1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 24
            direction = (-1.0, -1.0, -1.0)
            normal = (1.0, -2.0, 1.0)
            slip_systems.append((direction, normal))

        if '123' in slip_families:

            # system 25
            direction = (1.0, -1.0, -1.0)
            normal = (3.0, 1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 26
            direction = (-1.0, -1.0, 1.0)
            normal = (1.0, 2.0, 3.0)
            slip_systems.append((direction, normal))

            # system 27
            direction = (-1.0, 1.0, -1.0)
            normal = (2.0, 3.0, 1.0)
            slip_systems.append((direction, normal))

            # system 28
            direction = (-1.0, -1.0, 1.0)
            normal = (2.0, 1.0, 3.0)
            slip_systems.append((direction, normal))

            # system 29
            direction = (-1.0, 1.0, -1.0)
            normal = (1.0, 3.0, 2.0)
            slip_systems.append((direction, normal))

            # system 30
            direction = (1.0, -1.0, -1.0)
            normal = (3.0, 2.0, 1.0)
            slip_systems.append((direction, normal))

            # system 31
            direction = (-1.0, -1.0, -1.0)
            normal = (-3.0, 1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 32
            direction = (1.0, -1.0, 1.0)
            normal = (-1.0, 2.0, 3.0)
            slip_systems.append((direction, normal))

            # system 33
            direction = (1.0, 1.0, -1.0)
            normal = (-2.0, 3.0, 1.0)
            slip_systems.append((direction, normal))

            # system 34
            direction = (1.0, -1.0, 1.0)
            normal = (-2.0, 1.0, 3.0)
            slip_systems.append((direction, normal))

            # system 35
            direction = (1.0, 1.0, -1.0)
            normal = (-1.0, 3.0, 2.0)
            slip_systems.append((direction, normal))

            # system 36
            direction = (-1.0, -1.0, -1.0)
            normal = (-3.0, 2.0, 1.0)
            slip_systems.append((direction, normal))

            # system 37
            direction = (1.0, -1.0, 1.0)
            normal = (3.0, 1.0, -2.0)
            slip_systems.append((direction, normal))

            # system 38
            direction = (-1.0, -1.0, -1.0)
            normal = (1.0, 2.0, -3.0)
            slip_systems.append((direction, normal))

            # system 39
            direction = (-1.0, 1.0, 1.0)
            normal = (2.0, 3.0, -1.0)
            slip_systems.append((direction, normal))

            # system 40
            direction = (-1.0, -1.0, -1.0)
            normal = (2.0, 1.0, -3.0)
            slip_systems.append((direction, normal))

            # system 41
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, 3.0, -2.0)
            slip_systems.append((direction, normal))

            # system 42
            direction = (1.0, -1.0, 1.0)
            normal = (3.0, 2.0, -1.0)
            slip_systems.append((direction, normal))

            # system 43
            direction = (1.0, 1.0, -1.0)
            normal = (3.0, -1.0, 2.0)
            slip_systems.append((direction, normal))

            # system 44
            direction = (-1.0, 1.0, 1.0)
            normal = (1.0, -2.0, 3.0)
            slip_systems.append((direction, normal))

            # system 45
            direction = (-1.0, -1.0, -1.0)
            normal = (2.0, -3.0, 1.0)
            slip_systems.append((direction, normal))

            # system 46
            direction = (-1.0, 1.0, 1.0)
            normal = (2.0, -1.0, 3.0)
            slip_systems.append((direction, normal))

            # system 47
            direction = (-1.0, -1.0, -1.0)
            normal = (1.0, -3.0, 2.0)
            slip_systems.append((direction, normal))

            # system 48
            direction = (1.0, 1.0, -1.0)
            normal = (3.0, -2.0, 1.0)
            slip_systems.append((direction, normal))

    elif crystal_structure == 'hcp':

        SQRT3 = sqrt(3.)

        if 'basal' in slip_families:

            # system 1
            direction = (1.0, -SQRT3, 0.0)
            normal = (0.0, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 2
            direction = (1.0, SQRT3, 0.0)
            normal = (0.0, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 3
            direction = (-1.0, 0.0, 0.0)
            normal = (0.0, 0.0, 1.0)
            slip_systems.append((direction, normal))

        if 'prismatic' in slip_families:

            # system 4
            direction = (1.0, 0.0, 0.0)
            normal = (0.0, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 5
            direction = (1.0, SQRT3, 0.0)
            normal = (-SQRT3, 1.0, 0.0)
            slip_systems.append((direction, normal))

            # system 6
            direction = (-1.0, SQRT3, 0.0)
            normal = (-SQRT3, -1.0, 0.0)
            slip_systems.append((direction, normal))

            # Pyramidal A Slip Systems

            # system 7
            direction = (1.0, 0.0, 0.0)
            normal = (0.0, -2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 8
            direction = (1.0, SQRT3, 0.0)
            normal = (SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 9
            direction = (-1.0, SQRT3, 0.0)
            normal = (SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 10
            direction = (-1.0, 0.0, 0.0)
            normal = (0.0, 2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 11
            direction = (-1.0, -SQRT3, 0.0)
            normal = (-SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 12
            direction = (1.0, -SQRT3, 0.0)
            normal = (-SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

        if 'first order c+a' in slip_families:

            # system 13
            direction = (1.0, SQRT3, 2.0 * ratio_c_a)
            normal = (0.0, -2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 14
            direction = (-1.0, SQRT3, 2.0 * ratio_c_a)
            normal = (SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 15
            direction = (-1.0, 0.0, ratio_c_a)
            normal = (SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 16
            direction = (-1.0, -SQRT3, 2. * ratio_c_a)
            normal = (0.0, 2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 17
            direction = (1.0, -SQRT3, 2. * ratio_c_a)
            normal = (-SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 18
            direction = (1.0, 0.0, 2. * ratio_c_a)
            normal = (-SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 19
            direction = (-1.0, SQRT3, 2.0 * ratio_c_a)
            normal = (0.0, -2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 20
            direction = (-1.0, 0.0, ratio_c_a)
            normal = (SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 21
            direction = (-1.0, -SQRT3, 2. * ratio_c_a)
            normal = (SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 22
            direction = (1.0, -SQRT3, 2. * ratio_c_a)
            normal = (0.0, 2. * ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 23
            direction = (1.0, 0.0, ratio_c_a)
            normal = (-SQRT3 * ratio_c_a, ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

            # system 24
            direction = (1.0, SQRT3, 2. * ratio_c_a)
            normal = (-SQRT3 * ratio_c_a, -ratio_c_a, SQRT3)
            slip_systems.append((direction, normal))

        if 'second order c+a' in slip_families:

            # system 25
            direction = (-1.0, SQRT3, 2. * ratio_c_a)
            normal = (ratio_c_a, -SQRT3 * ratio_c_a, 2.0)
            slip_systems.append((direction, normal))

            # system 26
            direction = (-1.0, 0.0, ratio_c_a)
            normal = (ratio_c_a, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 27
            direction = (-1.0, -SQRT3, 2. * ratio_c_a)
            normal = (ratio_c_a, SQRT3 * ratio_c_a, 2.0)
            slip_systems.append((direction, normal))

            # system 28
            direction = (1.0, -SQRT3, 2. * ratio_c_a)
            normal = (-ratio_c_a, SQRT3 * ratio_c_a, 2.0)
            slip_systems.append((direction, normal))

            # system 29
            direction = (1.0, 0.0, ratio_c_a)
            normal = (-ratio_c_a, 0.0, 1.0)
            slip_systems.append((direction, normal))

            # system 30
            direction = (1.0, SQRT3, 2. * ratio_c_a)
            normal = (-ratio_c_a, -SQRT3 * ratio_c_a, 2.0)
            slip_systems.append((direction, normal))
    
    return slip_systems
