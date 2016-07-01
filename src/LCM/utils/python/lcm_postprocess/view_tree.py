#!/usr/bin/python

if __name__ == '__main__':

    import sys
    from _core import view_tree

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    view_tree(filename = name_file_input)

# end if __name__ == '__main__':