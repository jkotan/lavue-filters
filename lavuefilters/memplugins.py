# Copyright (C) 2017  DESY, Notkestr. 85, D-22607 Hamburg
#
# lavue is an image viewing program for photon science imaging detectors.
# Its usual application is as a live viewer using hidra as data source.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation in  version 2
# of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
# Authors:
#     Jan Kotanski <jan.kotanski@desy.de>
#

""" History memory plugins """

import numpy as np
import sys

if sys.version_info > (3,):
    unicode = str
else:
    bytes = str


class HistoryDump(object):

    """ History Dump"""

    def __init__(self, configuration=None):
        """ constructor

        :param configuration: a number of images
        :type configuration: :obj:`str`
        """
        try:
            self._maxindex = max(int(configuration), 2)
        except Exception:
            self._maxindex = 10

        #: (:class:`numpy.ndarray`) image stack
        self._imagestack = None
        self._current = 1
        self._lastimage = None
        self._first = True

    def initialize(self):
        """ initialize the filter
        """
        self._imagestack = None
        self._lastimage = None
        self._current = 1
        self._first = True

    def terminate(self):
        """ stop filter
        """
        self._imagestack = None
        self._lastimage = None
        self._current = 1

    def __call__(self, image, imagename, metadata, imagewg):
        """ call method

        :param image: numpy array with an image
        :type image: :class:`numpy.ndarray`
        :param imagename: image name
        :type imagename: :obj:`str`
        :param metadata: JSON dictionary with metadata
        :type metadata: :obj:`str`
        :param imagewg: image wigdet
        :type imagewg: :class:`lavuelib.imageWidget.ImageWidget`
        :returns: numpy array with an image
        :rtype: :class:`numpy.ndarray` or `None`
        """
        mdata = {}
        if self._lastimage is None or \
           not np.array_equal(self._lastimage, image):
            shape = image.shape
            dtype = image.dtype

            if self._imagestack is not None:
                if self._imagestack.shape[1:] != shape or \
                   self._imagestack.dtype != dtype:
                    self._imagestack = None

            if self._imagestack is None:
                newshape = np.concatenate(([self._maxindex], list(shape)))
                self._imagestack = np.zeros(dtype=dtype, shape=newshape)

            if self._current >= self._maxindex:
                self._current = 1
            lshape = len(self._imagestack.shape)
            if lshape == 3:
                self._imagestack[self._current, :, :] = image
                self._imagestack[0, :, :] = image
            elif lshape == 2:
                self._imagestack[self._current, :] = image
                self._imagestack[0, :] = image
            elif lshape == 1:
                self._imagestack[self._current] = image
                self._imagestack[0] = image

            self._current += 1
            self._lastimage = image

            if self._first:
                cblbl = {key: None for key in range(self._maxindex)}
            else:
                cblbl = {}
            mdata["channellabels"] = cblbl
            cblbl[0] = "0: the last image"
            cblbl[self._current - 1] = "%s: %s" % (
                self._current - 1, imagename)
            self._first = False

        if self._imagestack is not None:
            return (self._imagestack, mdata)

    def __del__(self):
        self.terminate()
