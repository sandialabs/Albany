#! /usr/bin/env python
"""

Description: Syncronize files between a repository directory and a working
project directory.
"""

__version__ = "1.0"
__author__  = "Bill Spotz"
__date__    = "23 Jun 2016"

# System imports
import argparse
import os
import sys

files = ["CTestConfig.cmake",
         "Project.xml",
         "albanyAeras",
         "ctest_nightly.cmake.frag",
         "gaia_modules.sh",
         "nightly_cron_script_albany_gaia.sh",
         "nightly_cron_script_trilinos_gaia.sh",
         "process_output.sh",
         "send_email.sh",
         "sync.py",
         "trilinosAeras"]

home = os.getenv("HOME")
default_repo_dir = os.path.join(home,
                                "Development",
                                "Albany",
                                "doc",
                                "dashboards",
                                "gaia.sandia.gov")

default_project_dir = os.path.join("/project",
                                   "projectdirs",
                                   "aeras",
                                   "nightlyGaiaCDash")

current_dir = os.getcwd()

########################################################################

def main(repo_dir, project_dir):
    for file in files:
        repo_file    = os.path.join(repo_dir   , file)
        project_file = os.path.join(project_dir, file)
        source       = None
        target       = None
        if os.path.isfile(repo_file):
            repo_mtime = os.path.getmtime(repo_file)
            if os.path.isfile(project_file):
                project_mtime = os.path.getmtime(project_file)
                if repo_mtime < project_mtime:
                    source = project_file
                    target = repo_file
                if repo_mtime > project_mtime:
                    source = repo_file
                    target = project_file
            else:
                source = repo_file
                target = project_file
        else:
            if os.path.isfile(project_file):
                source = project_file
                target = repo_file
            else:
                print("Warning: source for '%s' cannot be found" % file)
        if source and target:
            os.open(target,"w").write(os.open(source,"r").read())
            print("Copied '%s' to '%s'" % (source, target))

########################################################################

if __name__ == "__main__":

    # Construct the argument parser and parse the arguments
    progName = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(description=__doc__,
                                     version="%(prog)s " + __version__ +
                                     " " + __date__)
    parser.add_argument("-r", "--repo", action="store", dest="repo",
                        default="none", type=str, help="set the repository "
                        "directory")
    parser.add_argument("-p", "--project", action="store", dest="project",
                        default="none", type=str, help="set the project "
                        "directory")
    parser.add_argument("-l", "--list", action="store_true", dest="list",
                        default=False, help="list the files to be synced and "
                        "exit")
    parser.add_argument("-d", "--defaults", action="store_true",
                        dest="defaults", default=False, help="list the default "
                        "directories that will be used and exit")
    args = parser.parse_args()

    # Determine the repository directory
    if args.repo != "none":
        repo_dir = args.repo
    elif (current_dir.split(os.path.sep)[-4:] ==
          default_repo_dir.split(os.path.sep)[-4:]):
        repo_dir = current_dir
    else:
        repo_dir = default_repo_dir

    # Determine the project directory
    if args.project != "none":
        project_dir = args.project
    else:
        project_dir = default_project_dir

    # Perform the requested action
    if args.list:
        print("Files to be synced:")
        for file in files:
            print("    %s" % file)
    elif args.defaults:
        print("Repository directory = '%s'" % repo_dir   )
        print("Project directory    = '%s'" % project_dir)
    else:
        if not os.path.isdir(repo_dir):
            sys.exit("%s: Error: repository directory '%s' not found" %
                     (progName, repo_dir))
        if not os.path.isdir(project_dir):
            sys.exit("%s: Error: project directory '%s' not found" %
                     (progName, project_dir))
        main(repo_dir, project_dir)
