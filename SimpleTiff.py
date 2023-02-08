import requests
from io import BytesIO
from struct import unpack_from
import numpy as np
import math
from imagecodecs import jpeg2k_decode
from functools import lru_cache

def empty(*args):
    pass
if False: # set it to True for debugging
    debug = print
else:
    debug = empty

class FileReader:
    def __init__(self, fn):
        if fn.startswith("http://") or fn.startswith("https://"):
            self._remote = True
            self._name = fn
        else:  
            self._remote = False
            self._name = fn
            self.filehandle = open(fn, 'rb')
    @lru_cache
    def seek_and_read(self, offset, bytecount):
        if self._remote:
            data = BytesIO(requests.get(self._name, 
                    headers = {"Range": "bytes=%d-%d" % (offset, offset + bytecount-1)},
                    timeout=5)
                    .content).read()
            if len(data) != bytecount:
                raise ValueError('requests.get() failed with wrong number of bytes')                
        else: 
            self.filehandle.seek(offset)
            data = self.filehandle.read(bytecount)
        return data
    def close():
        if not self._remote:
            self.filehandle.close()

TYPE_DICT = {1:'BYTE', 2:'ASCII', 3:"SHORT", 4: "LONG", 5: 'RATIONAL',  6:'SBYTE', 7:'UNDEFINED', 8:'SSHORT', 9:'SLONG', 10: 'SRATIONAL', 11:'FLOAT', 12:'DOUBLE'}
TAG_DICT = {
    254:'NewSubfileType',
    255:'SubfileType',
    256:'ImageWidth',
    257:'ImageLength', 
    258:'BitsPerSample', 
    259:'Compression', #1: no compression; 2: CCITT Group 3 1-D Huffman RLE encoding; 32773: PackBits compression
    262:'PhotometricInterpretation', #3: Palette color
    263:'Threshholding',
    270:'ImageDescription',
    273:'StripOffsets',
    274:'Orientation',
    277:'SamplesPerPixel',
    278:'RowsPerStrip',
    279:'StripByteCounts',
    284:'PlanarConfiguration',
    320:'ColorMap',
    322:'TileWidth',
    323:'TileLength',
    324:'TileOffsets',
    325:'TileByteCounts',
    347:'JPEGTables',
    530:'YCbCrSubsampling',
    32997:'ImageDepth',
    34675:'ICC Profile',
}

class TiffPage:
    def __init__(self, offset, data, tiffFile):
        self.imagewidth = None
        self.imaglength = None
        self.dataoffsets = []
        self.databytecounts = []
        self._offset = offset
        self._next_offset = None
        self.parent = tiffFile
        self.parse_data(data)
    def parse_data(self, data):
        endian = self.parent._endian
        debug("-"*25 + str(self._offset) + " " + "0x%06x" % self._offset + "-"*25)
        # ret = f.seek(offset, 0)
        # num_d_entries = f.read(2)
        num_d_entries = unpack_from(endian + "H", data[:2])[0]
        debug(num_d_entries, "entries")
        assert(len(data) == 2 + num_d_entries * 12 + 4 )

        entries = [data[(2+i*12):(2+i*12 + 12)] for i in range(num_d_entries)]
        #next_offset = f.read(4)
        next_offset = data[-4:]
        self._next_offset = unpack_from(endian + "I", next_offset)[0]

        entries = [ (e[:2], e[2:4], e[4:8], e[8:12]) for e in entries] # tag (sorted), type, count, value/offset
        # unpack_from(endian + "h", entries[0][1])[0] 

        entries = [ (unpack_from(endian + "H", e[0])[0], # tag
                    TYPE_DICT[unpack_from(endian + "H", e[1])[0]], # type
                    unpack_from(endian + "I", e[2])[0], # count
                    e[3]) for e in entries # value/offset
                    ]  
        entriesDict = {}
        for idx, val in enumerate(entries):
            tag, type, count, value = val
            if type == 'SHORT':
                v = unpack_from(endian + "H", entries[idx][3])[0]
                o = unpack_from(endian + "I", entries[idx][3])[0]
            elif type == 'LONG': ## TODO: need to check tag to determine offset or value
                v = unpack_from(endian + "I", entries[idx][3])[0]
                o = v
            elif type == 'ASCII':
                o = unpack_from(endian + "I", entries[idx][3])[0]
                v = "!!Not yet read!!"
                # f.seek(o, 0)
                # v = f.read(entries[idx][2])
                # v = unpack_from(endian + str(entries[idx][2]) + "c", v)
                # v = v[:-1] # get ride of \0
                # v = ''.join([i.decode() for i in v])
            else:
                v = entries[idx][3]
                o = unpack_from(endian + "I", entries[idx][3])[0]
            entries[idx] = (str(tag) + ":" + TAG_DICT.get(tag, "***") , type, count, v, o)
            entriesDict[entries[idx][0]] = (type, count, v, o)

        
        for idx, val in enumerate(entries):
            tag, type, count, v, o = val
            val = tag, type, str(count), str(v), str(o)
            # print(val)
            debug('%32s\t%s\t%s\t%s\t%s' % val)

        debug("self._next_offset = %d 0x%06x" % (self._next_offset, self._next_offset))
        # assign common properties
        self.is_tiled = False
        self.imagewidth = entriesDict['256:ImageWidth'][2]
        self.imagelength = entriesDict['257:ImageLength'][2]
        if '322:TileWidth' in entriesDict:
            self.tilewidth = entriesDict['322:TileWidth'][2]
            self.is_tiled = True
        if '323:TileLength' in entriesDict:
            self.tilelength = entriesDict['323:TileLength'][2]
        self.imagedepth = entriesDict['32997:ImageDepth'][2]
        self.samplesperpixel = entriesDict['277:SamplesPerPixel'][2]
        if '324:TileOffsets' in entriesDict:
            self.dataoffsets = self.readArray(entriesDict['324:TileOffsets'])
        if '325:TileByteCounts' in entriesDict:
            self.databytecounts =  self.readArray(entriesDict['325:TileByteCounts'])
        self.dtype = np.uint8
    def readArray(self, val):
        (type, count, v, o) = val
        assert(type == 'LONG')
        data = self.parent.filehandle.seek_and_read(o, 4 * count)
        ret = unpack_from(self.parent._endian + str(count) + "I", data)
        return ret
        
class SimpleTiff:
    def __init__(self, name):
        self._name = name
        self.filehandle = FileReader(name)
        self.pages = []
        self._endian = None
        self.open()
    def open(self):
        # read header
        self.header = self.filehandle.seek_and_read(0, 8)
        if self.header[:2] == b'MM':
            debug("Big endian")
            endian = ">"
        elif self.header[:2] == b'II':
            debug("Little endian")
            endian = "<"
        else:
            raise ValueError('TIFF header is not recognized')
        self._endian = endian

        magic = self.header[2:4]
        assert(42 == unpack_from(endian+"H",magic)[0])

        offset = self.header[4:8]
        offset = unpack_from(endian + "I", offset)[0]

        # read IFDs
        data = self.read_ifd_block(offset)
        p = TiffPage(offset, data, self)
        self.pages.append(p)
        while p._next_offset != 0:
            data = self.read_ifd_block(p._next_offset)
            p = TiffPage(p._next_offset, data, self)
            self.pages.append(p)
            
    def close(self):
        self.filehandle.close()

    def read_ifd_block(self, offset):
        data = self.filehandle.seek_and_read(offset,  2 + 42 * 12 + 4) #IFD size is A+2+B*12 where A is IFD offset and B is # of entries, let's guess B = 42 at first attempt
        num_d_entries = unpack_from(self._endian + "H", data[:2])[0]
        if num_d_entries > 42:
            # read more bytes
            data2 = self.filehandle.seek_and_read(offset + 2 + 42 * 12 + 4, (num_d_entries - 42) * 12)
            data = data+data2
        else:
            data = data[: (2 + num_d_entries * 12 + 4)]
        return data
    # import numpy
    # # from tifffile import TiffFile
    # from SimpleTiff import SimpleTiff
    # import math
    # cache = {}
    def get_svs_tile(self, level, col, row):
        #if level not in cache:
        #    cache[level] = {}
        ## find the tiff image to work on

        ## find tiff tiles 
        ## extract images
        ## return results

        # return None
        # t = TiffFile(tiffFile)
        # t = SimpleTiff(tiffFile)
        t = self
        p0 = t.pages[0]
        p0_size = [p0.imagewidth, p0.imagelength]
        p3 = t.pages[3]
        p3_size = [p3.imagewidth, p3.imagelength]
        page  = [p for p in t.pages if math.fabs(p.imagewidth/p.imagelength - p0.imagewidth / p0.imagelength) < 0.01]
        page_size = [ (p.imagewidth, p.imagelength) for p in page ]
        s = (p0.imagewidth, p0.imagelength)
        dims = []
        while True:
            dims.append(s)
            i = (math.ceil(s[0]/2), math.ceil(s[1]/2))
            if i[0] == 1 and i[1] == 1: 
                break
            else:
                s = i
        dims.append(i)
        # level = 5
        # col = 1
        # row = 2
        ref_page_size = p3
        requested_dims = dims[-(level+1)]

        ref = [ p for p in page if p.imagewidth >= requested_dims[0]][-1]
        ref_size = [ref.imagewidth, ref.imagelength]
        scale = ref_size[0] / requested_dims[0]
        debug("scale = ", scale, ref_size[0], requested_dims[0])
        request_x = col 
        request_y = row
        page_x = min(request_x * 255 * scale, ref_size[0])
        page_y = min(request_y * 255 * scale, ref_size[1])
        page_width = min(255 * scale, ref_size[0] - page_x - 1)
        page_height = min(255 * scale, ref_size[1] - page_y - 1)
        (page_x, page_y, page_width, page_height) = map(int, (page_x, page_y, page_width, page_height))

        import read_region
        # ret = read_region.read_region(p0, 100, 200, 255, 255)
        debug("read_region:", ref_size, page_y, page_x, page_height, page_width )
        ret = self.read_region(ref, page_y, page_x, page_height, page_width, None) # TODO: add cache back cache[level])
        import cv2
        ret_s = cv2.resize(ret[0,:], (255, 255))
        _, JPEG = cv2.imencode('.jpeg', ret_s)
        return JPEG.tobytes()

    # def read_region(self, page, i0, j0, h, w, cache):
    # @profile
    def read_region(self, page, i0, j0, h, w, cache):
        """Extract a crop from a TIFF image file directory (IFD).
        
        Only the tiles englobing the crop area are loaded and not the whole page.
        This is usefull for large Whole slide images that can't fit int RAM.
        Parameters
        ----------
        page : TiffPage
            TIFF image file directory (IFD) from which the crop must be extracted.
        i0, j0: int
            Coordinates of the top left corner of the desired crop.
        h, w: int
            Desired crop height, width.
        Returns
        -------
        out : ndarray of shape (imagedepth, h, w, sampleperpixel)
            Extracted crop.
        """

        if not page.is_tiled:
            raise ValueError("Input page must be tiled.")

        im_width = page.imagewidth
        im_height = page.imagelength
        # debug("im dimension", im_width, im_height, " and request", i0, j0, h, w, )
        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
            debug(i0, h, im_height, " --- ", j0, w, im_width)
            raise ValueError("Requested crop area is out of image bounds.")

        tile_width, tile_height = page.tilewidth, page.tilelength
        i1, j1 = i0 + h, j0 + w

        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.empty((page.imagedepth,
                        (tile_i1 - tile_i0) * tile_height,
                        (tile_j1 - tile_j0) * tile_width,
                        page.samplesperpixel), dtype=page.dtype)
        fh = page.parent.filehandle
        # jpegtables = page.tags.get('JPEGTables', None)
        # if jpegtables is not None:
        #     jpegtables = jpegtables.value

        for i in range(tile_i0, tile_i1):
            for j in range(tile_j0, tile_j1):
                debug(i, j)
                index = int(i * tile_per_line + j)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                # if index in cache:
                #     data = cache[index]
                # else:
                #     #fh.seek(offset)
                #     #data = fh.read(bytecount)
                #     data = fh.seek_and_read(offset, bytecount)
                #     cache[index] = data
                data = fh.seek_and_read(offset, bytecount)
                debug('index offset: ', index, offset, bytecount, offset + bytecount)
                # tile , indices, shape = jpeg2k_decode(data) # jpegtables) #page.decode(data, index, jpegtables) 
                tile = jpeg2k_decode(data) # jpegtables) #page.decode(data, index, jpegtables) 

                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]

if __name__ == "__main__":
    pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
    urlPrefix = 'http://localhost:8000/'
    localTiffFile = pathPrefix + '10021.svs'
    netTiffFile = urlPrefix + '10021.svs'

    print(SimpleTiff(localTiffFile).header)
    print(SimpleTiff(netTiffFile).header)
    assert(SimpleTiff(localTiffFile).header == SimpleTiff(netTiffFile).header) 

    #import cProfile
    # cProfile.run('ret = SimpleTiff(localTiffFile).get_svs_tile(10, 0, 0)')
    ret = SimpleTiff(localTiffFile).get_svs_tile(10, 0, 0)
    fOut = open('test.jpeg', 'wb')
    fOut.write(ret)
    fOut.close()
