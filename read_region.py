from tifffile import TiffFile
import numpy as np
from imagecodecs import jpeg_decode
from imagecodecs import jpeg2k_decode
# jpeg2k_encode, jpeg2k_decode, jpeg2k_check, jpeg2k_version, JPEG2K
    
def read_region(page, i0, j0, h, w, cache):
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

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
        print(i0, h, im_height, " --- ", j0, w, im_width)
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
    print(page.dtype, type(page.dtype), "  dtype")
    jpegtables = page.tags.get('JPEGTables', None)
    if jpegtables is not None:
        jpegtables = jpegtables.value

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            if index in cache:
                data = cache[index]
            else:
                fh.seek(offset)
                data = fh.read(bytecount)
                cache[index] = data
            print('index offset: ', index, offset, bytecount, offset + bytecount)
            # tile , indices, shape = jpeg2k_decode(data) # jpegtables) #page.decode(data, index, jpegtables) 
            tile = jpeg2k_decode(data) # jpegtables) #page.decode(data, index, jpegtables) 

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    print("out dim = ", out.shape)
    return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]



if __name__ == "__main__": 
    # tiffFile = sys.argv[1]
    pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
    tiffFile = pathPrefix + '10021.svs'
    t = TiffFile(tiffFile)
    page = t.pages[0]
    i0, j0, h, w = 25000, 25000, 150, 150
    img = read_region(page, i0, j0, h, w)
    
    import numpy as np
    from PIL import Image as im
    # img = np.array([cmap[i] for i in idx])
    # img = img.reshape((entriesDict['257:ImageLength'][2], entriesDict['256:ImageWidth'][2],3))
    # img = img / 65535 * 255
    print(img.shape) # 8192 x 3
    imgH = im.fromarray(np.uint8(img), mode = 'RGB')
    imgH.show()