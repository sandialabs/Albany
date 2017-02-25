from albany_stats import *
from run_albany import run_albany

def create_outfile_name(options, n, M, N):
    s = 'albany_' 
    if options.dakota:
        s = s + 'dakota_'
    elif options.sg:
        s = s + 'sg_'
    if options.sg_kl:
        s = s + 'kl_'
    s = s + options.tag
    s = s + str(n)
    if options.dakota or options.sg:
        s = s + '_' + str(M) + '_' + str(N)
    s = s + '.out'
    return s

def create_dakota_outfile_name(options, n, M, N):
    return 'dakota_' + options.tag + str(n) + '_' + str(M) + '_' + str(N) + '.out'

def error_sweep(n_range, M_range, N_range, options):
    dakota = options.dakota
    sg = options.sg

    all_stats = []
    i = 0
    for n in n_range:
        
        all_stats.append([])

        # Run single deterministic problem
        options.dakota = False
        options.sg = False
        M = 5
        options.num_mesh = n
        options.num_kl = M
        options.input = options.input_det
        options.input_heat = options.input_heat_det
        options.input_neut = options.input_neut_det
        options.output = create_outfile_name(options, n, M, 0)
        det_stats = run_albany(options)
        t0 = det_stats.total[0]

        # Loop over dimensions
        for M in M_range:
            options.dakota = dakota
            options.sg = sg
            options.num_kl = M
            if M == 1:
                options.num_kl = M+1
            options.dim = 2*M
            options.num_jac_kl = M+1

            # Loop over orders, run problem, and get stats
            stats = AlbanyStats()
            if sg:
                stats = AlbanySGStats()
            elif dakota:
                stats = AlbanyDakotaStats()
            for N in N_range: 
                if sg:
                    options.input = options.input_sg
                    options.input_heat = options.input_heat_sg
                    options.input_neut = options.input_neut_sg
                elif dakota:
                    options.input = options.input_dakota
                options.order = N
                options.output = create_outfile_name(options, n, M, N)
                if dakota:
                    options.dakota_output = create_dakota_outfile_name(options,n,M,N)
                my_stats = run_albany(options)
                stats.extend_stats(my_stats)

            # Compute errors based on highest order
            l = len(stats.mean)
            stats.compute_errors(stats.mean[l-1], stats.var[l-1])

            # Normalize times
            stats.normalize_times(t0)

            all_stats[i].append(stats)

        i = i+1

    return all_stats

def dim_sweep(n_range, M_range, N_range, options):
    dakota = options.dakota
    sg = options.sg

    all_stats = []
    i = 0
    for n in n_range:
        
        all_stats.append([])

        # Run single deterministic problem
        options.dakota = False
        options.sg = False
        M = 5
        options.num_mesh = n
        options.num_kl = M
        options.input = options.input_det
        options.output = create_outfile_name(options, n, M, 0)
        det_stats = run_albany(options)
        t0 = det_stats.total[0]

        # Loop over orders
        for N in N_range: 
            options.order = N
            options.dakota = dakota
            options.sg = sg
            if sg:
                options.input = options.input_sg
            elif dakota:
                options.input = options.input_dakota

            # Loop over dimensions, run problem, and get stats
            stats = AlbanyStats()
            if sg:
                stats = AlbanySGStats()
            elif dakota:
                stats = AlbanyDakotaStats()
            for M in M_range:
                options.num_kl = M
                if M == 1:
                    options.num_kl = M+1
                options.dim = M
                options.num_jac_kl = M+1
                options.output = create_outfile_name(options, n, M, N)
                if dakota:
                    options.dakota_output = create_dakota_outfile_name(options,n,M,N)
                my_stats = run_albany(options)
                stats.extend_stats(my_stats)

            # Normalize times
            stats.normalize_times(t0)

            all_stats[i].append(stats)

        i = i+1

    return all_stats


