from flask import Flask, abort, make_response, render_template, url_for
from flask_cors import CORS, cross_origin
import io
from optparse import OptionParser
import requests


pathPrefix = '/Users/zhanxw/Downloads/test.tiff/'
tiffFile = pathPrefix + '10021.svs'
import numpy
from SimpleTiff import SimpleTiff
import math

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

DZI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpeg"
  Overlap="1"
  TileSize="254"
  >
  <Size
    Height="%d"
    Width="%d"
  />
</Image>
"""
@app.route('/proxy/<path:path>.dzi')
@cross_origin()
def proxy_dzi(path):
    print("proxy: " + path)
    fn = urlPrefix + '10021.svs'
    print(fn)
    o=SimpleTiff(fn)
    xml = DZI_TEMPLATE % (o.pages[0].imagelength, o.pages[0].imagewidth)
    resp = make_response(xml)
    resp.mimetype = 'application/text'
    return resp

@app.route('/proxy/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
@cross_origin()
def proxy_tile(path, level, col, row, format):
    print("proxy request: ", path, level, col, row, format)

    # fn = pathPrefix + '10021.dzi_files/%d/%d_%d.%s' % (level, col, row, format)
    fn = pathPrefix + '10021.svs'
    o = SimpleTiff(fn)
    resp = make_response(o.get_svs_tile(level, col, row))
    resp.mimetype = 'image/jpeg'
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