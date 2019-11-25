#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""
Args:
    :f: A function to use the filename.
    :filename: System filename path.
"""
import os
def get_function_file(f, filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        return f(filename)

