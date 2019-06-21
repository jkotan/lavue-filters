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
import sys
import time
import pytz
import datetime
import os
import socket

if sys.version_info > (3,):
    unicode = str
else:
    bytes = str


class H5PYdump(object):

    """ H5PY image writer plugin"""

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
        self._maxindex = maxindex

        #: (:obj: `str` >) list of indexes for gap
        self.__filename = configuration \
            if configuration \
            else "/tmp/lavueh5pydump.nxs"
        sfn = self.__filename.split(".")
        if len(sfn) > 1:
            self.__oldfilename = ".".join(sfn[:-1]) + ".1." + sfn[-1]
        else:
            self.__oldfilename = ".".join(sfn[:-1]) + ".1"

        #: (:obj:`int`) group index
        self.__grpindex = 1
        #: (:obj:`int`) image index
        self._imgindex = 0
        #: (:obj:`h5py._hl.file.File`) root object
        self._h5file = None
        #: (:obj:`h5py._hl.group.Group`) NXentry group
        self._h5entry = None
        #: (:obj:`h5py._hl.group.Group`) NXdata group
        self._h5data = None
        #: (:obj:`h5py._hl.dataset.Dataset`) data field
        self._h5field = None
        #: (:obj:`h5py._hl.dataset.Dataset`) data_name field
        self._h5field_name = None

    def initialize(self):
        """ initialize the filter
        """
        self._reset()

    def terminate(self):
        """ stop filter
        """
        if hasattr(self._h5file, "close"):
            if self._h5entry is not None:
                self._h5entry.create_dataset(
                    "end_time", shape=[1], chunks=None,
                    dtype=h5py.special_dtype(vlen=unicode),
                    maxshape=(None,), data=unicode(self._currenttime()))
            self._h5file.close()
            self._h5file = None

    @classmethod
    def _currenttime(cls):
        """ returns current time string

        :returns: current time
        :rtype: :obj:`str`
        """
        tzone = time.tzname[0]
        tz = pytz.timezone(tzone)
        fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
        starttime = tz.localize(datetime.datetime.now())
        return str(starttime.strftime(fmt))

    def _create_file(self):
        """ creates a new file
        """
        if os.path.isfile(self.__filename):
            os.rename(self.__filename, self.__oldfilename)
        self._h5file = h5py.File(self.__filename, 'w', libver='latest')
        self._h5file.attrs["file_time"] = unicode(self._currenttime())
        self._h5file.attrs["HDF5_version"] = u""
        self._h5file.attrs["NX_class"] = u"NXroot"
        self._h5file.attrs["NeXus_version"] = u"4.3.0"
        self._h5file.attrs["file_name"] = unicode(self.__filename)
        self._h5file.attrs["file_update_time"] = unicode(self._currenttime())

    def _create_entry(self):
        """ add a new scan entry
        """
        self.__grpindex += 1
        self._h5entry = self._h5file.create_group(
            "scan_%s" % self.__grpindex)
        self._h5entry.attrs["NX_class"] = "NXentry"
        self._h5data = self._h5entry.create_group("data")
        self._h5data.attrs["NX_class"] = "NXdata"
        self._h5field = None
        self._h5field_name = None
        starttime = self._currenttime()
        self._h5entry.create_dataset(
            "start_time", shape=[1], chunks=None,
            dtype=h5py.special_dtype(vlen=unicode),
            maxshape=(None,), data=unicode(starttime))
        self._h5entry.create_dataset(
            "title", shape=[1], chunks=None,
            dtype=h5py.special_dtype(vlen=unicode),
            maxshape=(None,), data=unicode("LaVue NeXus dump"))
        self._h5entry.create_dataset(
            "experiment_identifier", shape=[1], chunks=None,
            dtype=h5py.special_dtype(vlen=unicode),
            maxshape=(None,),
            data=unicode("%s@%s" % (starttime, socket.gethostname())))

    def _reopen(self):
        """  reopen the file
        """
        self._h5file.close()
        self._h5file = h5py.File(self.__filename, 'r+', libver='latest')
        self._h5file.swmr_mode = True
        self._h5entry = self._h5file.get("scan_%s" % self.__grpindex)
        self._h5data = self._h5entry.get("data")
        self._h5field = self._h5data.get("data")
        self._h5field_name = self._h5data.get("data_name")

    def _reset(self):
        """  remove the file and create a new one
        """
        if hasattr(self._h5file, "close"):
            self._h5file.close()
        self._create_file()
        self._h5entry = None
        self._h5data = None
        self._h5field = None
        self._h5field_name = None
        self._imgindex = 0
        self.__grpindex -= 1

    def _check_shape_and_dtype(self, image):
        """ reset the field if its shape or type has changed

        :param image: numpy array with an image
        :type image: :class:`numpy.ndarray`
        """
        if self._h5field is not None:
            if self._h5field.shape[1:] != image.shape or \
               self._h5field.dtype != image.dtype:
                self._h5field = None
                if self._h5entry is not None:
                    self._h5entry.create_dataset(
                        "end_time", shape=[1], chunks=None,
                        dtype=h5py.special_dtype(vlen=unicode),
                        maxshape=(None,), data=unicode(self._currenttime()))

    def _create_data_field(self, data):
        """ create a data field

        :param data: variable with data
        :type data: :class:`numpy.ndarray`
        """
        shape = [0]
        shape.extend(data.shape)
        chunk = [1]
        chunk.extend(data.shape)
        dtype = data.dtype
        maxshape = [None] * len(shape)
        self._h5field = self._h5data.create_dataset(
            "data",
            shape=tuple(shape),
            dtype=dtype,
            chunks=tuple(chunk),
            compression="gzip",
            compression_opts=2,
            maxshape=tuple(maxshape))

    def _create_scalar_field(self, name, dtype=None):
        """ create a string field

        :param name: field name
        :type name: :obj:`str`
        :param dtype: field data type
        :type dtype: :obj:`str`
        :returns: field object
        :rtype dtype: :class:`h5py._hl.dataset.Dataset`
        """
        dtype = dtype or h5py.special_dtype(vlen=unicode)
        field = self._h5data.create_dataset(
            name,
            shape=(0,),
            chunks=(1,),
            dtype=h5py.special_dtype(vlen=unicode),
            maxshape=(None,))
        return field

    def _append_data(self, field, data):
        """ appends the data

        :param field: field object
        :type field: :class:`h5py._hl.dataset.Dataset`
        :param data: variable with data
        :type data: :class:`numpy.ndarray`
        """
        new_shape = list(field.shape)
        new_shape[0] += 1
        field.resize(tuple(new_shape))
        if len(new_shape) == 1:
            field[-1] = data
        elif len(new_shape) == 1:
            field[-1, :] = data
        elif len(new_shape) == 3:
            field[-1, :, :] = data
        elif len(new_shape) == 4:
            field[-1, :, :, :] = data
        if hasattr(self._h5field, "flush"):
            field.flush()

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
        if len(image.shape) > 1:
            image = np.swapaxes(image, 0, 1)
        self._imgindex += 1
        if self._maxindex > 0 and self._imgindex >= self._maxindex:
            self._reset()
        self._check_shape_and_dtype(image)
        if self._h5field is None:
            self._create_entry()
            self._create_data_field(image)
            self._h5field_name = self._create_scalar_field("data_name")
            self._reopen()
        self._append_data(self._h5field, image)
        self._append_data(self._h5field_name, unicode(imagename))

    def __del__(self):
        self.terminate()


class H5PYdumpdiff(H5PYdump):

    """ H5PY image writer plugin when image differs from previous"""

    def __init__(self, configuration=None):
        """ constructor

        :param configuration: file name to dump images and, max image numbers
        :type configuration: :obj:`str`
        """
        H5PYdump.__init__(self, configuration)
        self.__lastimage = None

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
        if len(image.shape) > 1:
            image = np.swapaxes(image, 0, 1)
        if self.__lastimage is None or \
           not np.array_equal(self.__lastimage, image):
            self._imgindex += 1
            if self._maxindex > 0 and self._imgindex >= self._maxindex:
                self._reset()
            self._check_shape_and_dtype(image)
            if self._h5field is None:
                self._create_entry()
                self._create_data_field(image)
                self._h5field_name = self._create_scalar_field("data_name")
                self._reopen()
            self._append_data(self._h5field, image)
            self._append_data(self._h5field_name, unicode(imagename))
        self.__lastimage = np.array(image)
