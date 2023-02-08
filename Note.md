Introduction
============

This script is to proxy a SVS as a DZI file for openseadragon.

Quick start
===========

1. Start the local proxy: `python app.py -p 9000`
2. Modify `tileSources` in the `openseadragon` script.
   Serve SVS to openseadragon as a DZI file. 
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

    Note: here we use `http://localhost:8000` by running: `python byte.server.py`. This is CORS-enabled, byte-range enabled simple HTTP server. 


Reference
=========

jpstroop/dzi_to_iiif.py
https://gist.github.com/jpstroop/4624253

The DZI File Format
https://github.com/openseadragon/openseadragon/wiki/The-DZI-File-Format
