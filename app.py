from flask import Flask, abort, make_response, render_template, url_for, request
from flask_cors import CORS, cross_origin
from optparse import OptionParser
from SimpleTiff import SimpleTiff

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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
    print("proxy_dzi: " + path)
    o=SimpleTiff(path)
    xml = DZI_TEMPLATE % (o.pages[0].imagelength, o.pages[0].imagewidth)
    resp = make_response(xml)
    resp.mimetype = 'application/text'
    return resp

@app.route('/proxy/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
@cross_origin()
def proxy_tile(path, level, col, row, format):
    print("proxy_tile: ", path, level, col, row, format)
    o = SimpleTiff(path)
    resp = make_response(o.get_svs_tile(level, col, row))
    resp.mimetype = 'image/jpeg'
    return resp

@app.route('/proxy/svs/dummy.dzi')
def proxy_svs_dzi():
    """Serve SVS to openseadragon as a DZI file. 
       In the openseadragon `tileSources`, use a fake DZI file `dummy.dzi` and arguments
          `file` for local SVS files and
          'url` for remote SVS files. 
    
    e.g., to serve a SVS, "abc/abc.svs", specify the following:

            OpenSeadragon({
            id:            "example-xmlhttprequest-for-dzi",
            tileSources:   "http://localhost:9000/proxy/svs/dummy.dzi?file=abc%2Fabc.svs",
        });
    to serve a SVS, "http://localhost:8000/abc/abc.svs", specify the following:

            OpenSeadragon({
            id:            "example-xmlhttprequest-for-dzi",
            tileSources:   "http://localhost:9000/proxy/svs/dummy.dzi?url=http%3A%2F%2Flocalhost%3A8000%2Fabc%2Fabc.svs",
        });

    Test in curl: 
      curl 'http://localhost:9000/proxy/svs/dummy.dzi?file=abc%2Fabc.svs'
    Returns
    -------
    the content of .dzi file
    """
    fn = request.args.get('file', None)
    url = request.args.get('url', None)
    if fn:
        print('serve fn: ', fn)
        return proxy_dzi(fn)
    if url:
        print('serve url: ', url)
        return proxy_dzi(url)

@app.route('/proxy/svs/dummy_files/<int:level>/<int:col>_<int:row>.<format>')
def proxy_svs_tile(level, col, row, format):
    fn = request.args.get('file', None)
    url = request.args.get('url', None)
    if fn:
        print('serve fn')
        return proxy_tile(fn, level, col, row, format)
    if url:
        print('serve url')
        return proxy_tile(url, level, col, row, format)

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