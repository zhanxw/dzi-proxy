# jello.tif	256x192 8-bit RGB (lzw palette) Paul Heckbert "jello"
fn = "libtiffpic/jello.tif"
import sys
if len(sys.argv) > 1:
    fn = sys.argv[1]

f = open(fn, 'rb')
from struct import *
header = f.read(2)
print(header)
if header == b'MM':
    print("Big endian")
    endian = ">"
elif header == b'II':
    print("Little endian")
    endian = "<"
else:
    print("something wrong")

magic = f.read(2)
print(magic)
import struct
[ord(i) for i in struct.unpack_from(endian+"cc",magic)]  ## should be [0, 42]

offset = f.read(4)
offset = struct.unpack_from(endian + "i", offset)[0]

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

while offset != 0:
    print("-"*25 + str(offset) + " " + "0x%06x" % offset + "-"*25)
    ret = f.seek(offset, 0)
    num_d_entries = f.read(2)
    num_d_entries = struct.unpack_from(endian + "H", num_d_entries)[0]
    print(num_d_entries, "entries")

    entries = [f.read(12) for i in range(num_d_entries)]
    next_offset = f.read(4)
    next_offset = struct.unpack_from(endian + "I", next_offset)[0]

    entries = [ (e[:2], e[2:4], e[4:8], e[8:12]) for e in entries] # tag (sorted), type, count, value/offset
    # struct.unpack_from(endian + "h", entries[0][1])[0] 

    entries = [ (struct.unpack_from(endian + "H", e[0])[0], # tag
                TYPE_DICT[struct.unpack_from(endian + "H", e[1])[0]], # type
                struct.unpack_from(endian + "I", e[2])[0], # count
                e[3]) for e in entries # value/offset
                ]  
    entriesDict = {}
    for idx, val in enumerate(entries):
        tag, type, count, value = val
        if type == 'SHORT':
            v = struct.unpack_from(endian + "H", entries[idx][3])[0]
            o = struct.unpack_from(endian + "I", entries[idx][3])[0]
        elif type == 'LONG': ## TODO: need to check tag to determine offset or value
            v = struct.unpack_from(endian + "I", entries[idx][3])[0]
            o = v
        elif type == 'ASCII':
            o = struct.unpack_from(endian + "I", entries[idx][3])[0]
            f.seek(o, 0)
            v = f.read(entries[idx][2])
            v = struct.unpack_from(endian + str(entries[idx][2]) + "c", v)
            v = v[:-1] # get ride of \0
            v = ''.join([i.decode() for i in v])
        else:
            v = entries[idx][3]
            o = struct.unpack_from(endian + "I", entries[idx][3])[0]
        entries[idx] = (str(tag) + ":" + TAG_DICT.get(tag, "***") , type, count, v, o)
        entriesDict[entries[idx][0]] = (type, count, v, o)

    
    for idx, val in enumerate(entries):
        tag, type, count, v, o = val
        val = tag, type, str(count), str(v), str(o)
        # print(val)
        print('%32s\t%s\t%s\t%s\t%s' % val)

    print("next_offset = %d 0x%06x" % (next_offset, next_offset))

    offset = next_offset

# f.close()

def getStripInfo(entries):
    offset = []
    bytes = []
    for e in entries:
        if e[0] == '273:StripOffsets':
            if e[2] > 1: ## likely  e[3] is offset instead of value
                f.seek(e[3])
                for i in range(e[2]):
                    assert(e[1] == 'LONG')
                    v = f.read(4)
                    offset.append(struct.unpack_from(endian + "I", v)[0])
            else:
                offset.append(e[3])
        if e[0] == '279:StripByteCounts':
            if e[2] > 1: ## likely e[3] is offset instead of value
                f.seek(e[3])
                for i in range(e[2]):
                    assert(e[1] == 'LONG')
                    v = f.read(4)
                    bytes.append(struct.unpack_from(endian + "I", v)[0])    
            else:
                bytes.append(e[3])
    return (offset, bytes)
def getStrips(offsets, bytes):
    res = []
    for o, b in zip(offsets, bytes):
        f.seek(o)
        res.append(f.read(b))
    return res

# from:
# https://github.com/psd-tools/packbits/blob/master/LICENSE.txt
# https://github.com/psd-tools/packbits/blob/master/src/packbits.py
def decode(data):
    """
    Decodes a PackBit encoded data.
    """
    data = bytearray(data) # <- python 2/3 compatibility fix
    result = bytearray()
    pos = 0
    while pos < len(data):
        header_byte = data[pos]
        if header_byte > 127:
            header_byte -= 256
        pos += 1
        if 0 <= header_byte <= 127:
            result.extend(data[pos:pos+header_byte+1])
            pos += header_byte+1
        elif header_byte == -128:
            pass
        else:
            result.extend([data[pos]] * (1 - header_byte))
            pos += 1
    #return bytes(result)
    return result

## 262:PhotometricInterpretation
##   0: WhiteIsZero
##   1: BlackIsZero
##   2: RGB
##   3: Palette color
##   4: Transparency Mask
## for palette color
if entriesDict.get('262:PhotometricInterpretation')[2] == 3: ## Palette color
    (offsets, bytes) = getStripInfo(entries)
    res = getStrips(offsets, bytes)
    res2 = [decode(d) for d in res]
    [len(r) for r in res2]
    [max(r) for r in res2]
    [min(r) for r in res2]
    # read colormap
    # f.seek(45600)
    type, count, value, offset = entriesDict.get('320:ColorMap')
    print(entriesDict.get('320:ColorMap'))
    f.seek(offset)
    cmap = f.read(count * 2) ## read by 16-bit
    cmap = struct.unpack_from(endian + str(count) + "H", cmap)
    cmap = [(cmap[i], cmap[count//3 + i ], cmap[count//3*2 + i]) for i in range(count//3)]

    ## 
    idx = []
    for r in res2:
        for rr in r:
            idx.append(rr)
    import numpy as np
    from PIL import Image as im
    img = np.array([cmap[i] for i in idx])
    img = img.reshape((entriesDict['257:ImageLength'][2], entriesDict['256:ImageWidth'][2],3))
    img = img / 65535 * 255
    print(img.shape) # 8192 x 3
    imgH = im.fromarray(np.uint8(img), mode = 'RGB')
    imgH.show()

if entriesDict.get('262:PhotometricInterpretation')[2] == 2: ## RGB
    if entriesDict.get('259:Compression')[2] == 7: ## JPEG compression
        print("JPEG")
        (offsets, bytes) = getStripInfo(entries)
        res = getStrips(offsets, bytes)
       
        # below is to examine JPEG contents
        data = res[0]
        from struct import unpack
        marker_mapping = { \
            0xffd8: "Start of Image", \
            0xffe0: "Application Default Header", \
            0xffdb: "Quantization Table", \
            0xffc0: "Start of Frame", \
            0xffc4: "Define Huffman Table",\
            0xffda: "Start of Scan",\
            0xffd9: "End of Image"\
        }
        while(True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                print('return')
                break
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break       

        ## below let's try to decode JPEG
        
        # import imagecodecs
        # print('\n'.join(dir(imagecodecs)))
        from imagecodecs import jpeg_decode
        data = res[0]
        import numpy as np
        from PIL import Image as im
        if '258:BitsPerSample' in entriesDict:
            bitspersample = 2 ** entriesDict.get('258:BitsPerSample')[1]
        else:
            bitspersample = None
        if '347:JPEGTables' in entriesDict:
            type, count, value, offset = entriesDict.get('347:JPEGTables')
            f.seek(offset)
            jpegTable = f.read(count)
        else:
            jpegTable = None
        img = jpeg_decode(data, bitspersample=bitspersample, tables=jpegTable)
        # img = img.reshape((entriesDict['257:ImageLength'][2], entriesDict['256:ImageWidth'][2],3))
        # img = img / 65535 * 255
        # print(img.shape) # 8192 x 3
        imgH = im.fromarray(np.uint8(img), mode = 'RGB')
        imgH.show()