#!/usr/bin/env python

import re
import sys
import math

class AlbanyStats:
    def __init__(self):
        self.total = []
        self.residual = []
        self.jacobian = []
        self.cijk = []
        self.solve = []
        self.mean = []
        self.var = []
        self.mean_err = []
        self.var_err = []
        self.re_float = re.compile('([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)')
        self.total_string = r'Albany: \*\*\*Total Time\*\*\*.*'
        self.residual_string = '> Albany Fill: Residual.*'
        self.jacobian_string = '> Albany Fill: Jacobian.*'
            

    def add_stats(self, total, residual, jacobian, cijk, solve, mean, std_dev):
        self.total.append(total)
        self.residual.append(residual)
        self.jacobian.append(jacobian)
        self.cijk.append(cijk)
        self.solve.append(solve)
        self.mean.append(mean)
        sd = []
        for j in range(len(std_dev)):
            sd.append(std_dev[j]*std_dev[j])
        self.var.append(sd)

    def extend_stats(self, stats):
        self.total.extend(stats.total)
        self.residual.extend(stats.residual)
        self.jacobian.extend(stats.jacobian)
        self.cijk.extend(stats.cijk)
        self.solve.extend(stats.solve)
        self.mean.extend(stats.mean)
        self.var.extend(stats.var)

    def get_stats_from_file(self, fname):
        f = open(fname, 'r')
        total_time = 0
        residual_time = 0
        jacobian_time = 0
        cijk_time = 0
        solve_time = 0
        mean = []
        std_dev = []
        skip = 0
        for line in f:
            m = re.search(self.total_string, line)
            if m != None:
                total_time = self.get_float(m.group(0))

            m = re.search(self.residual_string, line)
            if m != None:
                residual_time = self.get_float(m.group(0))

            m = re.search(self.jacobian_string, line)
            if m != None:
                jacobian_time = self.get_float(m.group(0))

            m = re.search('Total Triple-Product Tensor Fill Time.*', line)
            if m != None:
                cijk_time = self.get_float(m.group(0))

            m = re.search('NOX: Total Linear Solve Time.*', line)
            if m != None:
                solve_time = self.get_float(m.group(0))

            # Get mean which is 2 lines after 'Mean = ...'
            if (skip == 3):
                m = re.match(r'\s+\d+\s+\d+\s+(.*)', line)
                mean.append(self.get_float(m.group(1)))
                skip = 0
            if (skip == 2):
                m = re.match(r'\s+\d+\s+\d+\s+(.*)', line)
                mean.append(self.get_float(m.group(1)))
                skip = 3
            if (skip == 1):
                skip = 2
            if (re.search('Mean:', line) != None):
                skip = 1

            # Get std_dev which is 2 lines after 'Standard Deviation = ...'
            if (skip == 13):
                m = re.match(r'\s+\d+\s+\d+\s+(.*)', line)
                std_dev.append(self.get_float(m.group(1)))
                skip = 0
            if (skip == 12):
                m = re.match(r'\s+\d+\s+\d+\s+(.*)', line)
                std_dev.append(self.get_float(m.group(1)))
                skip = 13
            if (skip == 11):
                skip = 12
            if (re.search('Std. Dev.:', line) != None):
                skip = 11

        return total_time, residual_time, jacobian_time, cijk_time, solve_time, mean, std_dev

    def add_stats_from_file(self, fname):
        total_time,residual_time,jacobian_time,cijk_time,solve_time,me,sd = self.get_stats_from_file(fname)
        self.add_stats(total=total_time,
                       residual=residual_time,
                       jacobian=jacobian_time,
                       cijk=cijk_time,
                       solve=solve_time,
                       mean=me,
                       std_dev=sd)

    def add_stats_from_files(self, fnames):
        for fname in fnames:
            self.add_stats_from_file(fname)

    def compute_errors(self, mean_true, var_true):
        self.mean_err = []
        self.var_err = []
        n = len(self.mean)
        for i in range(n):
            m_err = []
            v_err = []
            m = len(self.mean[i])
            for j in range(m):
                m_err.append(math.fabs((self.mean[i][j] - mean_true[j]) / mean_true[j]))
                v_err.append(math.fabs((self.var[i][j] - var_true[j]) / var_true[j]))
            self.mean_err.append(m_err)
            self.var_err.append(v_err)

    def normalize_times(self, t):
        n = len(self.total)
        for i in range(n):
            self.total[i] = self.total[i] / t
            self.residual[i] = self.residual[i] / t
            self.jacobian[i] = self.jacobian[i] / t
            self.cijk[i] = self.cijk[i] / t
            self.solve[i] = self.solve[i] / t

    def __str__(self):
        n = len(self.total)
        p = 2
        w = p + 6
        m = len(self.mean[0])
        s = ''
        sep = '  '
        s = s + '{0:^{width}}'.format('Total', width=w) + sep
        s = s + '{0:^{width}}'.format('Solve', width=w) + sep
        s = s + '{0:^{width}}'.format('Res', width=w) + sep
        s = s + '{0:^{width}}'.format('Jac', width=w) + sep
        s = s + '{0:^{width}}'.format('Cijk', width=w) + sep
        for i in range(m):
            s = s + '{0:^{width}}'.format('Mean'+str(i+1), width=w) + sep
        for i in range(m):
            s = s + '{0:^{width}}'.format('Var'+str(i+1), width=w) + sep
        if (len(self.mean_err) > 0):
            for i in range(m):
                s = s + '{0:^{width}}'.format('Mean Err'+str(i+1), width=w) + sep
            for i in range(m):
                s = s + '{0:^{width}}'.format('Var Err'+str(i+1), width=w) + sep
        s = s + '\n'
        for i in range(n):
            s = s + '{0:>{width}.{precision}e}'.format(self.total[i], width=w, precision=p) + sep
            s = s + '{0:>{width}.{precision}e}'.format(self.solve[i], width=w, precision=p) + sep
            s = s + '{0:>{width}.{precision}e}'.format(self.residual[i], width=w, precision=p) + sep
            s = s + '{0:>{width}.{precision}e}'.format(self.jacobian[i], width=w, precision=p) + sep
            s = s + '{0:>{width}.{precision}e}'.format(self.cijk[i], width=w, precision=p) + sep
            for j in range(m):
                s = s + '{0:>{width}.{precision}e}'.format(self.mean[i][j], width=w, precision=p) + sep
            for j in range(m):
                s = s + '{0:>{width}.{precision}e}'.format(self.var[i][j], width=w, precision=p) + sep
            if (len(self.mean_err) > 0):
                for j in range(m):
                    s = s + '{0:>{width}.{precision}e}'.format(self.mean_err[i][j], width=w, precision=p) + sep
                for j in range(m):
                    s = s + '{0:>{width}.{precision}e}'.format(self.var_err[i][j], width=w, precision=p) + sep
            s = s + '\n'
        return s

    def get_float(self, str):
        return float(self.re_float.search(str).group(0))

class AlbanySGStats(AlbanyStats):
    def __init__(self):
        AlbanyStats.__init__(self)
        self.total_string = r'AlbanySG: \*\*\*Total Time\*\*\*.*'
        self.residual_string = '> Albany Fill: SGResidual.*'
        self.jacobian_string = '> Albany Fill: SGJacobian.*'

class AlbanySGNIStats(AlbanyStats):
    def __init__(self):
        AlbanyStats.__init__(self)
        self.total_string = r'AlbanySG: \*\*\*Total Time\*\*\*.*'
        self.residual_string = '> Albany Fill: Residual.*'
        self.jacobian_string = '> Albany Fill: Jacobian.*'

class AlbanyMPNIStats(AlbanyStats):
    def __init__(self):
        AlbanyStats.__init__(self)
        self.total_string = r'AlbanySG: \*\*\*Total Time\*\*\*.*'
        self.residual_string = '> Albany Fill: MPResidual.*'
        self.jacobian_string = '> Albany Fill: MPJacobian.*'

class AlbanyDakotaStats(AlbanyStats):
    def __init__(self):
        AlbanyStats.__init__(self)
        self.total_string = r'AlbanyDakota: \*\*\*Total Time\*\*\*.*'

    def add_stats_from_file(self, albany_fname, dakota_fname):
        total_time,residual_time,jacobian_time,cijk_time,solve_time,me,sd = self.get_stats_from_file(albany_fname)

        # Get mean and standard deviation
        f = open(dakota_fname, 'r')
        for line in f:
            m = re.match('\s+expansion:\s+(.*)', line)
            if m != None:
                s = m.group(1).split()
                me = self.get_float(s[0])
                sd = self.get_float(s[1])

        self.add_stats(total=total_time,
                       residual=residual_time,
                       jacobian=jacobian_time,
                       cijk=cijk_time,
                       solve=solve_time,
                       mean=me,
                       std_dev=sd)


def get_stats(fnames):
    stats = AlbanyStats()
    stats.add_stats_from_files(fnames)
    
    return stats

if __name__ == "__main__":
    stats = get_stats(sys.argv[1:])
    print stats
