import requests
import re
import numbers
from io import BytesIO
from struct import unpack_from
import numpy as np
import math
from imagecodecs import jpeg2k_decode, jpeg_decode
from functools import lru_cache
from PIL import Image

from utils.deepzoom import DeepZoomGenerator

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
                debug('requests.get() failed with wrong number of bytes: request [ %d ] bytes, received [ %d ] bytes' % (bytecount, len(data)))                
        else: 
            self.filehandle.seek(offset)
            data = self.filehandle.read(bytecount)
        return data
    def close(self):
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

PHOTOMETRIC_SAMPLES = {
    0: 1,  # MINISWHITE
    1: 1,  # MINISBLACK
    2: 3,  # RGB
    3: 1,  # PALETTE
    4: 1,  # MASK
    5: 4,  # SEPARATED
    6: 3,  # YCBCR
    8: 3,  # CIELAB
    9: 3,  # ICCLAB
    10: 3,  # ITULAB
    32803: 1,  # CFA
    32844: 1,  # LOGL ?
    32845: 3,  # LOGLUV
    34892: 3,  # LINEAR_RAW ?
    51177: 1,  # DEPTH_MAP ?
    52527: 1,  # SEMANTIC_MASK ?
}


class TiffPage:
    def __init__(self, offset, data, tiffFile):
        self.imagewidth = None
        self.imagelength = None
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
        else:
            self.tilewidth = 0
        if '323:TileLength' in entriesDict:
            self.tilelength = entriesDict['323:TileLength'][2]
        else:
            self.tilelength = 0
        if '32997:ImageDepth' in entriesDict:
            self.imagedepth = entriesDict['32997:ImageDepth'][2]
        self.samplesperpixel = entriesDict['277:SamplesPerPixel'][2]
        if '324:TileOffsets' in entriesDict:
            self.dataoffsets = self.readArray(entriesDict['324:TileOffsets'])
        if '325:TileByteCounts' in entriesDict:
            self.databytecounts =  self.readArray(entriesDict['325:TileByteCounts'])
        if '259:Compression' in entriesDict:
            self.compression = entriesDict['259:Compression'][2]
        if '347:JPEGTables' in entriesDict:
            self.jpegTable = self.readBytes(entriesDict['347:JPEGTables'])
        else:
            self.jpegTable = None
        if '262:PhotometricInterpretation' in entriesDict:
            self.photometric = entriesDict['262:PhotometricInterpretation'][2]
        else:
            self.photometric = None  
        if '270:ImageDescription' in entriesDict:
            self.description = entriesDict['270:ImageDescription'][2]
        else:
            self.description = ''
        self.dtype = np.uint8
        self.shape = (self.imagelength, self.imagewidth, PHOTOMETRIC_SAMPLES[self.photometric])
        self.ndim = len(self.shape)

    def readArray(self, val):
        (type, count, v, o) = val
        assert(type == 'LONG')
        data = self.parent.filehandle.seek_and_read(o, 4 * count)
        ret = unpack_from(self.parent._endian + str(count) + "I", data)
        return ret

    def readBytes(self, val): ## this is used for reading JPEGTables
        (type, count, v, o) = val
        assert(type == 'UNDEFINED')
        ret = self.parent.filehandle.seek_and_read(o, count)
        return ret 


class ImageTiles(object):
    """ Generate image tiles with in a given region or load existing tiles.
        Always call image_tiles.load_tiles() first before access other functions.
        rois return tile parameters: [x0, y0, w, h].
        coords return padded parameters: [x0, y0, w, h] in raw image, require padding.
        pad_width return pad width to fill image with patch_size, require padding.
    """
    def __init__(self, image_size, patch_size, padding=None, box=None):
        if isinstance(image_size, numbers.Number):
            image_size = (image_size, image_size)
        self.image_size = image_size
        
        if isinstance(patch_size, numbers.Number):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        
        if isinstance(padding, numbers.Number):
            padding = (padding, padding)
        self.padding = padding
        
        w, h = self.image_size
        if box is None:
            x0, y0, x1, y1 = [0, 0, w, h]
        else:
            x0, y0 = max(box[0], 0), max(box[1], 0)
            x1, y1 = min(box[2], w), min(box[3], h)
        self.box = [x0, y0, x1, y1]
        self.shape = None
    
    def load_tiles(self, tiles=None):
        # Calculate x_t and y_t
        if tiles is not None:
            self.x_t, self.y_t = tiles[:,0], tiles[:,1]
        else:
            x0, y0, x1, y1 = self.box
            w_p, h_p = self.patch_size
            self.y_t, self.x_t = np.mgrid[y0:y1:h_p, x0:x1:w_p]
        self.shape = self.x_t.shape

        return self
    
    def rois(self):
        x0, y0, x1, y1 = self.box
        w_p, h_p = self.patch_size
        
        h_t, w_t = np.minimum(h_p, y1 - self.y_t), np.minimum(w_p, x1 - self.x_t)
        # h_t, w_t = (y1 - self.y_t).clip(max=h_p), (x1 - self.x_t).clip(max=w_p)
        return np.stack([self.x_t, self.y_t, w_t, h_t], -1)
    
    def coords(self, padding=None):
        w, h = self.image_size
        w_p, h_p = self.patch_size
        w_d, h_d = padding or self.padding
        
        # we use (0, 0) instead of (x0, y0) to pad with original image
        x_s, y_s = (self.x_t - w_d).clip(0), (self.y_t - h_d).clip(0)
        w_s, h_s = np.minimum(self.x_t + w_p + w_d, w) - x_s, np.minimum(self.y_t + h_p + h_d, h) - y_s
        
        return np.stack([x_s, y_s, w_s, h_s], axis=-1)
    
    def pad_width(self, padding=None):
        w, h = self.image_size
        w_p, h_p = self.patch_size
        w_d, h_d = padding or self.padding
        
        pad_l, pad_u = (w_d - self.x_t).clip(0), (h_d - self.y_t).clip(0)
        pad_r, pad_d = (self.x_t + w_p + w_d - w).clip(0), (self.y_t + h_p + h_d - h).clip(0)
        
        return np.stack([pad_l, pad_r, pad_u, pad_d], axis=-1)


class SimpleTiff:
    def __init__(self, name):
        self._name = name
        self.filehandle = FileReader(name)
        self.pages = []
        self._endian = None
        self.open()
        self.register_entries()

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
    
    def register_entries(self, verbose=1):
        self.magnitude = None
        self.mpp = None
        self.description = None
        self.page_indices = []
        self.level_dims = []
        self.level_downsamples = []

        slide = self
        self.description = slide.pages[0].description

        # magnification
        print(self.description)
        val = re.findall(r'\|((?i:AppMag)|(?i:magnitude)) = (?P<mag>[\d.]+)', self.description)
        self.magnitude = float(val[0][1]) if val else None
        if verbose and self.magnitude is None:
            print(f"Didn't find magnitude in description.")

        # mpp
        val = re.findall(r'\|((?i:MPP)) = (?P<mpp>[\d.]+)', self.description)
        self.mpp = float(val[0][1]) if val else None
        if verbose and self.mpp is None:
            print(f"Didn't find mpp in description.")

        ## level_dims consistent with open_slide: (w, h), (OriginalHeight, OriginalWidth)
        level_dims, scales, page_indices = [(slide.pages[0].shape[1], slide.pages[0].shape[0])], [1.0], [0]
        for page_idx, page in enumerate(slide.pages[1:], 1):
            if 'label' in page.description or 'macro' in page.description:
                continue
            if page.tilewidth == 0 or page.tilelength == 0:
                continue
            h, w = page.shape[0], page.shape[1]
            if round(level_dims[0][0]/w) == round(level_dims[0][1]/h):
                level_dims.append((w, h))
                scales.append(level_dims[0][0]/w)
                page_indices.append(page_idx)

        order = sorted(range(len(scales)), key=lambda x: scales[x])
        self.page_indices = [page_indices[idx] for idx in order]
        self.level_dims = [level_dims[idx] for idx in order]
        self.level_downsamples = [scales[idx] for idx in order]
        self.n_levels = len(self.level_downsamples)

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
    
    @property
    def level_dimensions(self):
        return tuple(self.level_dims)
    
    def info(self):
        return {
            'magnitude': self.magnitude,
            'mpp': self.mpp,
            'level_dims': self.level_dims,
            'description': self.description,
        }

    def get_scales(self, x):
        """ x: (w, h) image_size tuple or a page index. """
        if isinstance(x, numbers.Number):
            w, h = self.level_dims[x]
        else:
            w, h = x
        
        return (w, h), [np.array([w/_[0], h/_[1]]) for _ in self.level_dims]

    def get_resize_level(self, x=None, downsample_only=False, epsilon=1e-2):
        """ Get nearest page level index for a given image_size/factor.
            x: (w, h) tuple or a downsampled scale_factor (.
            downsample_only: only pick the 
        """
        if isinstance(x, numbers.Number):
            factor = x
        else:
            w, h = x
            factor = min(self.level_dims[0][0]/w, self.level_dims[0][1]/h)
        rel_scales = np.array([d / factor for d in self.level_downsamples])
        
        if downsample_only:
            assert factor >= 1, f"Factor={factor}, cannot be downsampled."
            return np.where(rel_scales <= 1 + epsilon)[0][-1]
        else:
            return np.abs(np.log(rel_scales)).argmin()

    def deepzoom_coords(self, patch_size, padding=0, image_size=0, box=None):
        """ Generate tile coordinates.
            patch_size: patch_size of int or (patch_width, patch_height).
            page: the page index or image_size.
        """
        (w, h), scales = self.get_scales(image_size)
        tiles = ImageTiles((w, h), patch_size=patch_size, padding=padding, box=box)
        tiles.load_tiles()
        
        return tiles.coords()

    def get_patch(self, x, level=0):        
        x0, y0, w, h = x
        # tifffile don't reorder page, so need convertion here. Little bit slow.
        tiff_page_idx = self.page_indices[level] % len(self.pages)
        patch = self.read_region(self.pages[tiff_page_idx], y0, x0, h, w)
        patch = Image.fromarray(patch[0])

        return patch

    # @profile
    def read_region(self, page, i0, j0, h, w, cache=None):
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

        tile_width, tile_height = page.tilewidth, page.tilelength
        i1, j1 = i0 + h, j0 + w
        i0, j0 = max(0, i0), max(0, j0)
        i1, j1 = min(i0 + h, im_height), min(j0 + w, im_width)

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
                if page.compression == 33003:
                    tile = jpeg2k_decode(data) # , tables = page.jpegTable) # jpegtables) #page.decode(data, index, jpegtables) 
                elif page.compression == 7:
                    # refer to tifffile.py  jpeg_decode_colorspace()
                    # 2 means RGB
                    if page.photometric == 2:
                        colorspace = 2
                    else:
                        colorspace = None
                    outcolorspace = 2    
                    tile = jpeg_decode(data, tables = page.jpegTable, colorspace=colorspace, outcolorspace=outcolorspace,) # jpegtables) #page.decode(data, index, jpegtables) 
                else:
                    debug("A suitable decoder is not specified")
                    tile = jpeg2k_decode(data) # jpegtables) #page.decode(data, index, jpegtables) 
                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]


if __name__ == "__main__":
    def test():
        pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
        urlPrefix = 'http://localhost:8000/'
        localTiffFile = pathPrefix + '10021.svs'
        netTiffFile = urlPrefix + '10021.svs'

        print(SimpleTiff(localTiffFile).header)
        print(SimpleTiff(netTiffFile).header)
        assert(SimpleTiff(localTiffFile).header == SimpleTiff(netTiffFile).header) 

    import sys
    def usage():
        print("Usage:")
        print("\tpython SimpleTiff.py in.svs level col row out.jpeg")
    if len(sys.argv) != 6:
        usage()
        sys.exit(1)
    
    scriptFile, tiffFile, level, col, row, outFn = sys.argv
    level, col, row = map(int, (level, col, row))
    ret = SimpleTiff(tiffFile).get_svs_tile(level, col, row)
    fOut = open(outFn, 'wb')
    fOut.write(ret)
    fOut.close()
    print("[ %s ] level [ %d ] col [ %d ] row [ %d ] is converted into [ %s ]" % (tiffFile, level, col, row, outFn))
    sys.exit(0) 
