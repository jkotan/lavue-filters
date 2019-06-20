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

""" NeXus h5py writer plugins """

import h5py
import numpy as np

# import json
# from scipy import ndimage


class H5PYdump(object):

    """ Horizontal gap filter"""

    def __init__(self, configuration=None):
        """ constructor

        :param configuration: file name to dump images and,  max image numbers
        :type configuration: :obj:`str`
        """
        try:
            sconf = configuration.split(",")
            configuration = sconf[0]
            maxindex = int(sconf[1])
        except Exception:
            maxindex = 1000

        #: (:obj: `int`) maximal image number
        self.__maxindex = maxindex

        #: (:obj: `str` >) list of indexes for gap
        self.__filename = configuration \
            if configuration \
            else "/tmp/lavueh5pydump.h5"

        self.__grpindex = 0
        self.__imgindex = 0
        self.__h5entry = None
        self.__h5data = None
        self.__h5field = None

        self.__h5file = h5py.File(self.__filename, 'w', libver='latest')

    def _add_new_entry(self):
        """ add a new scan entry
        """
        self.__grpindex += 1
        self.__h5entry = self.__h5file.create_group(
            "scan_%s" % self.__grpindex)
        self.__h5entry.attrs["NX_class"] = "NXentry"
        self.__h5data = self.__h5entry.create_group("data")
        self.__h5data.attrs["NX_class"] = "NXdata"
        self.__h5field = None

    def _reopen(self):
        """  reopen the file
        """
        self.__h5file.close()
        self.__h5file = h5py.File(self.__filename, 'r+', libver='latest')
        self.__h5file.swmr_mode = True
        self.__h5entry = self.__h5file.get("scan_%s" % self.__grpindex)
        self.__h5data = self.__h5entry.get("data")
        self.__h5field = self.__h5data.get("data")

    def _reset(self):
        """  remove the file and create a new one
        """
        self.__h5file.close()
        self.__h5file = h5py.File(self.__filename, 'w', libver='latest')
        self.__h5entry = None
        self.__h5data = None
        self.__h5field = None
        self.__imgindex = 0

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
        self.__imgindex += 1
        if self.__maxindex > 0 and self.__imgindex >= self.__maxindex:
            self._reset()
        if self.__h5field is not None:
            if self.__h5field.shape[1:] != image.shape or \
               self.__h5field.dtype != image.dtype:
                self.__h5field = None
        if self.__h5field is None:
            self._add_new_entry()
            shape = [0]
            shape.extend(image.shape)
            chunk = [1]
            chunk.extend(image.shape)
            dtype = image.dtype
            maxshape = [None] * len(shape)
            self.__h5field = self.__h5data.create_dataset(
                "data",
                shape=tuple(shape),
                chunks=tuple(chunk),
                dtype=dtype,
                maxshape=tuple(maxshape))
            self._reopen()

        new_shape = list(self.__h5field.shape)
        new_shape[0] += 1
        self.__h5field.resize(tuple(new_shape))
        self.__h5field[-1, :, :] = np.transpose(image)
        if hasattr(self.__h5field, "flush"):
            self.__h5field.flush()
