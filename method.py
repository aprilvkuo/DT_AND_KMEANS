#!/usr/bin/env python
# encoding: utf-8

"""
@author: Aprilvkuo
@file: method.py
@time: 16-11-24 下午1:45
"""

import csv
import os
import xlrd



if __name__ == '__main__':
    name = 'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls'
    data = xlrd.open_workbook(name)