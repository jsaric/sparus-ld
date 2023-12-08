"""
TPS file format I/O. Adapted from https://gitlab.com/ryan-feather/py_tps
"""

from enum import Enum
import numpy as np
import io
import typing

TPS_SEP = ' '
LINE_SEP = '='
END_LINE = '\n'

class TPSKeywords(Enum):
    LM = 'LM'
    CURVES = 'CURVES'
    POINTS = 'POINTS'
    IMAGE = "IMAGE"
    ID = 'ID'
    SCALE = 'SCALE'
    COMMENT = 'COMMENT'


def read_tps_coordinates(io_stream: io.IOBase, n_lines: int) -> np.ndarray:
    """ Read n lines and convert them to a numpy 2d array with img rows being column 0 and img column being column 1.

    :param io_stream: an open io stream to read from
    :param n_lines: number of lines to read
    :return:
    """
    result = []
    text_lines = [io_stream.readline() for _ in range(n_lines)]
    for line in text_lines:
        result.append([float(coord) for coord in line.split(TPS_SEP)])

    return np.asarray(result, dtype=float)


def write_tps_coordinates(io_stream: io.IOBase, coordinates: np.ndarray):
    """ Write the coordinatess to an I/O stream.

    :param io_stream:
    :param coordinates: nx2 numpy array
    :return:
    """
    if coordinates.shape[0]>0: # empty
        if coordinates.shape[1] != 2:
            raise ValueError('Coordinates is expected to be an nx2 array.')

        lines = []
        for row in range(coordinates.shape[0]):
            lines.append(TPS_SEP.join(coordinates[row].astype(str).tolist())+END_LINE)
        io_stream.writelines(lines)


def _get_int_value(base_string: str, value: str) -> int:
    """ Get the count value after the prefix in a string.

    :param base_string:
    :param value:
    :return:
    """

    return int(_get_post_value(base_string, value))


def _get_post_value(base_string: str, value: str) -> str:
    base_length = len(base_string)
    post = value[base_length:].strip()
    return post

def _make_post_value(base_string: str, value: object) ->str:
    return f'{base_string}{value}'

def _get_float_value(base_string: str, value: str) -> float:
    """ Get the number value after the prefix in a string.

    :param base_string:
    :param value:
    :return:
    """

    return float(_get_post_value(base_string, value))


def _line_start(value):
    return f'{value}{LINE_SEP}'


class TPSPoints:
    """ Simple readable container for points.

    """

    def __init__(self, points: np.ndarray):
        self.points = points
        self.length = points.shape[0]

    @classmethod
    def read_stream(cls, io_stream: io.IOBase, n_lines: int):
        """ Read IO stream for n_lines landmarks.  Expects format like

            1872.00000 1876.00000
            1915.00000 1966.00000
            1958.00000 2055.00000
            1940.00000 2138.00000

        :param io_stream:
        :param n_lines:
        :return: TPSPoints object
        """
        points = read_tps_coordinates(io_stream, n_lines)
        return TPSPoints(points)

    def write_to_stream(self, io_stream: io.IOBase):
        """ Write the points to the stream.

        :param io_stream:
        :return:
        """
        write_tps_coordinates(io_stream, self.points)


class TPSCurve:
    """ Simple readable container for a curve definition

    ::param tps_points: TPSPoints.
    """
    _POINTS_LINE_START = _line_start(TPSKeywords.POINTS.value)

    def __init__(self, tps_points: TPSPoints):
        self.tps_points = tps_points

    @classmethod
    def read_stream(cls, io_stream: io.IOBase):
        """ Read TPS Curve from a stream. Expects format like

            POINTS=4
            1872.00000 1876.00000
            1915.00000 1966.00000
            1958.00000 2055.00000
            1940.00000 2138.00000


        :param io_stream:
        :return:
        """

        points_line = io_stream.readline()
        if not points_line.startswith(cls._POINTS_LINE_START):
            raise NotImplementedError(
                f'Expect first line in CURVE definition to be {cls._POINTS_LINE_START}, got {points_line}')

        n_points = _get_int_value(cls._POINTS_LINE_START, points_line)
        tps_points = TPSPoints.read_stream(io_stream, n_points)
        return TPSCurve(tps_points)

    def write_to_stream(self, io_stream: io.IOBase):
        """ Write the curve to the stream.

        :param io_stream:
        :return:
        """
        io_stream.writelines([_make_post_value(self._POINTS_LINE_START, self.tps_points.points.shape[0])+END_LINE])
        self.tps_points.write_to_stream(io_stream)


class TPSImage:
    _LM_LINE_START = _line_start(TPSKeywords.LM.value)
    _CURVES_LINE_START = _line_start(TPSKeywords.CURVES.value)
    _IMAGE_LINE_START = _line_start(TPSKeywords.IMAGE.value)
    _ID_LINE_START = _line_start(TPSKeywords.ID.value)
    _SCALE_LINE_START = _line_start(TPSKeywords.SCALE.value)
    _COMMENT_LINE_START = _line_start(TPSKeywords.COMMENT.value)

    def __init__(self, image: str, landmarks: TPSPoints = None, curves: typing.List[TPSCurve] = None,
                 id_number: str = None, comment: str = None, scale: float = None):
        self.image = image
        self.landmarks = landmarks or TPSPoints(np.zeros(0))
        self.curves = curves
        self.id_number = id_number
        self.comment = comment
        self.scale = scale

    @classmethod
    def read_stream(cls, io_stream: io.IOBase):
        """ Reverse engineered reading a LM from a stream. Expects image to start with LM.

        :param io_stream:
        :return:
        """
        first_line = io_stream.readline().strip()
        if not first_line.startswith(cls._LM_LINE_START):
            raise NotImplementedError(
                f'Expect first line in IMAGE definition to be {cls._LM_LINE_START}, got {first_line}')

        n_points = _get_int_value(cls._LM_LINE_START, first_line)
        # always seems to start with landmarks, even if 0 length
        landmarks = TPSPoints.read_stream(io_stream, n_points)
        curves = None
        image = None
        id_number = None
        comment = None
        scale = None

        # optionally
        # CURVES=2
        # IMAGE=
        # ID=0
        # COMMENT = Oxyrhopus_melanogenys
        # SCALE=0.004585

        stream_pos = None
        next_line = io_stream.readline().strip()
        # will always be at least an IMAGE, tag, so this is always safe to do at least once
        eof = False
        while not eof and not next_line.startswith(cls._LM_LINE_START):

            if next_line.startswith(cls._CURVES_LINE_START):
                n_curves = _get_int_value(cls._CURVES_LINE_START, next_line)
                curves = [TPSCurve.read_stream(io_stream) for _ in range(n_curves)]

            elif next_line.startswith(cls._IMAGE_LINE_START):
                image = _get_post_value(cls._IMAGE_LINE_START, next_line)

            elif next_line.startswith(cls._ID_LINE_START):
                id_number = _get_post_value(cls._ID_LINE_START, next_line)

            elif next_line.startswith(cls._COMMENT_LINE_START):
                comment = _get_post_value(cls._COMMENT_LINE_START, next_line)

            elif next_line.startswith(cls._SCALE_LINE_START):
                scale = _get_float_value(cls._SCALE_LINE_START, next_line)

            stream_pos = io_stream.tell()
            next_line = io_stream.readline()
            eof = len(next_line)==0

        if not eof:  # then encountered a new image
            io_stream.seek(stream_pos)

        tps_image = TPSImage(image=image, landmarks=landmarks, curves=curves, id_number=id_number, comment=comment,
                             scale=scale)

        return tps_image

    def write_to_stream(self, io_stream: io.IOBase):
        """ Write the TPS Image to the stream.

        :param io_stream:
        :return:
        """

        # write points first if any
        io_stream.writelines([_make_post_value(self._LM_LINE_START, self.landmarks.points.shape[0])+END_LINE])
        self.landmarks.write_to_stream(io_stream)
        if self.curves is not None:
            io_stream.writelines([_make_post_value(self._CURVES_LINE_START, len(self.curves))+END_LINE])
            for curve in self.curves:
                curve.write_to_stream(io_stream)

        if self.image is not None: # not sure this is possible, but maintaining symmetry
            io_stream.writelines([_make_post_value(self._IMAGE_LINE_START, self.image)+END_LINE])

        if self.id_number is not None:
            io_stream.writelines(([_make_post_value(self._ID_LINE_START, self.id_number)+END_LINE]))

        if self.comment is not None:
            io_stream.writelines(([_make_post_value(self._COMMENT_LINE_START, self.comment)+END_LINE]))

        if self.scale is not None:
            io_stream.writelines(([_make_post_value(self._SCALE_LINE_START, self.scale)+END_LINE]))

        # end with blank line
        io_stream.writelines([END_LINE])


class TPSFile:
    """ Basic reader of images in a TPS File from tpsDIG

    """

    def __init__(self, images: typing.List[TPSImage]):
        self.images = images

    @classmethod
    def read_file(cls, path):
        with open(path,'r') as tps_stream:
            images = []
            position = 0
            line = tps_stream.readline()
            while len(line)>0:
                tps_stream.seek(position)
                images.append(TPSImage.read_stream(tps_stream))
                position = tps_stream.tell()
                line = tps_stream.readline()

        return TPSFile(images=images)

    def write_to_file(self, path):
        """ Write TPS Images

        :param path:
        :return:
        """
        with open(path, 'w') as tps_stream:
            for image in self.images:
                image.write_to_stream(tps_stream)
