#!/usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, os, time, george
import rvs_custom as rvs
from uncertainties import unumpy as unp
import astropy.io.fits as fits
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
from PyAstronomy.pyasl import broadGaussFast, rotBroad
