#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:35:42 2021

@author: bailez
"""

import geopandas as gpd

df = gpd.read_file('areas_dsei.shp')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

point = Point(-35, -7)
polygon = df[df['dsei']=='POTIGUARA']['geometry'].values[0]
print(polygon.contains(point))
