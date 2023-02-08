from flask import Flask, abort, make_response, render_template, url_for
from flask_cors import CORS, cross_origin
import io
from optparse import OptionParser
import requests

pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
urlPrefix = 'http://localhost:8000/'
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/<path:path>.dzi')
@cross_origin()
def dzi(path):
    
    resp = make_response(open(pathPrefix + '10021.dzi.dzi').read())
    resp.mimetype = 'application/text'
    return resp


@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
@cross_origin()
def tile(path, level, col, row, format):
    print("request: ", path, level, col, row, format)
    fn = pathPrefix + '10021.dzi_files/%d/%d_%d.%s' % (level, col, row, format)
    resp = make_response(open(fn, 'rb').read())
    resp.mimetype = 'image/jpeg'
    return resp

@app.route('/proxy/<path:path>.dzi')
@cross_origin()
def proxy_dzi(path):
    print("proxy: " + path)
    resp = make_response(requests.get(urlPrefix + path + ".dzi").text)
    resp.mimetype = 'application/text'
    return resp

pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
tiffFile = pathPrefix + '10021.svs'

import numpy
# from tifffile import TiffFile
from SimpleTiff import SimpleTiff
import math
cache = {}
def get_svs_tile(level, col, row):
    if level not in cache:
        cache[level] = {}
    ## find the tiff image to work on

    ## find tiff tiles 
    ## extract images
    ## return results

    # return None
    # t = TiffFile(tiffFile)
    t = SimpleTiff(tiffFile)
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
    print("scale = ", scale, ref_size[0], requested_dims[0])
    request_x = col 
    request_y = row
    page_x = min(request_x * 255 * scale, ref_size[0])
    page_y = min(request_y * 255 * scale, ref_size[1])
    page_width = min(255 * scale, ref_size[0] - page_x - 1)
    page_height = min(255 * scale, ref_size[1] - page_y - 1)
    (page_x, page_y, page_width, page_height) = map(int, (page_x, page_y, page_width, page_height))

    import read_region
    # ret = read_region.read_region(p0, 100, 200, 255, 255)
    print("read_region:", ref_size, page_y, page_x, page_height, page_width )
    ret = read_region.read_region(ref, page_y, page_x, page_height, page_width, cache[level])
    import cv2
    ret_s = cv2.resize(ret[0,:], (255, 255))
    _, JPEG = cv2.imencode('.jpeg', ret_s)
    return JPEG.tobytes()

@app.route('/proxy/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
@cross_origin()
def proxy_tile(path, level, col, row, format):
    print("proxy request: ", path, level, col, row, format)

    fn = pathPrefix + '10021.dzi_files/%d/%d_%d.%s' % (level, col, row, format)

    resp = make_response(get_svs_tile(level, col, row))
    resp.mimetype = 'image/jpeg'
    return resp


    slide, masks = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
        if masks is not None:
            mask = masks.get_tile(level, (col, row))
        else:
            if level == slide._dz_levels-1 and app.model is not None:
                mask = run_tile(tile, app.model, app.dataset_configs, mpp=None)
                mask = Image.fromarray(mask)
            else:
                mask = None
    except ValueError:
        # Invalid level or coordinates
        abort(404)

    buf = BytesIO()
    if mask is not None:
        tile.paste(mask, mask=mask.split()[-1])
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
    parser.add_option(
        '-l',
        '--listen',
        metavar='ADDRESS',
        dest='host',
        default='0.0.0.0',
        help='address to listen on [127.0.0.1]',
    )
    parser.add_option(
        '-p',
        '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=5000,
        help='port to listen on [5000]',
    )
    (opts, args) = parser.parse_args()
    app.run(host=opts.host, port=opts.port, threaded=True)    