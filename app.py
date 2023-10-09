from flask import Flask, abort, make_response, render_template, url_for, request
from flask_cors import CORS, cross_origin
from optparse import OptionParser
from SimpleTiff import SimpleTiff
from utils.deepzoom import DeepZoomGenerator
from functools import lru_cache
from io import BytesIO

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# from pydantic import BaseSettings
class Settings(object):
    slide_cache_size: int = 10

    # deepzoom_format: str = "jpeg"
    deepzoom_tile_size: int = 254
    deepzoom_overlap: int = 1
    deepzoom_limit_bounds: bool = True
    deepzoom_tile_quality: int = 75

settings = Settings()


class _SlideCache:
    def __init__(self, dz_opts):
        self.dz_opts = dz_opts

    @lru_cache(maxsize=settings.slide_cache_size, typed=False)
    def get(self, slide_path):
        osr = SimpleTiff(slide_path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        
        return slide


def _get_slide(path):
    try:
        slide = app.cache.get(path)
        return slide
    except:
        abort(404)


@app.before_first_request
def _setup():
    opts = {
        'tile_size': settings.deepzoom_tile_size, 
        'overlap': settings.deepzoom_overlap,
        'limit_bounds': settings.deepzoom_limit_bounds,
    }
    app.cache = _SlideCache(opts)


@app.route('/proxy/<path:path>.dzi')
@cross_origin()
def proxy_dzi(path):
    print("proxy_dzi: " + path)
    slide = _get_slide(path)
    if path.endswith('.tiff'):
        deepzoom_format = 'png'
    else:
        deepzoom_format = 'jpeg'

    resp = make_response(slide.get_dzi(deepzoom_format))
    resp.mimetype = 'application/xml'

    return resp


@app.route('/proxy/<path:path>.params')
@cross_origin()
def proxy_params(path):
    print("proxy_params: " + path)
    slide = _get_slide(path)
    info = slide._osr.info()
    params = {
        'slide_mpp': info['mpp'],
        'magnitude': info['magnitude'],
        'description': info['description'],
    }

    resp = make_response(params)
    resp.mimetype = 'json'
    print(resp)

    return resp


@app.route('/proxy/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
@cross_origin()
def proxy_tile(path, level, col, row, format):
    print("proxy_tile: ", path, level, col, row, format)
    slide = _get_slide(path)

    try:
        tile = slide.get_tile(level, (col, row), format=format)
    except ValueError:
        abort(404)  # Invalid level or coordinates

    buf = BytesIO()
    tile.save(buf, format, quality=settings.deepzoom_tile_quality)
    resp = make_response(buf.getvalue())
    resp.mimetype = f"image/{format}"

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


@app.route('/proxy/svs/params', methods=['GET', 'POST'])
def proxy_svs_params():
    print("we got here!!!!!!")
    fn = request.args.get('file', None)
    url = request.args.get('url', None)
    if fn:
        print('serve fn: ', fn)
        return proxy_params(fn)
    if url:
        print('serve url: ', url)
        return proxy_params(url)


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
        help='address to listen on [0.0.0.0]',
    )
    parser.add_option(
        '-p',
        '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=9000,
        help='port to listen on [9000]',
    )

    (opts, args) = parser.parse_args()
    ## imagecodec is not thread-safe, will cause problem in production server.
    app.run(host=opts.host, port=opts.port, threaded=False, processes=8)
