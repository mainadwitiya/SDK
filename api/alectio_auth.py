import requests
import webbrowser
import os
import json
from os import path
from uuid import uuid4
from webbrowser import open_new
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, HTTPError
from urllib.parse import urlparse, parse_qs
from requests.auth import HTTPBasicAuth
from flask import jsonify

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        self.wfile.write("All done. You can close the window now.".encode())
        code = urlparse(self.path).query.split("=")[1]
        if code:
            try:
                data = {
                    'grant_type':'authorization_code',
                    'code':code
                    }
                response = requests.post('http://127.0.0.1:5000/oauth/token', 
                                        auth=HTTPBasicAuth(client_id, client_secret), data=data)
                print(response.json())
                with open('client_token.json', 'w') as fp:
                    json.dump(response.json(), fp)

            except:
                print("SDK Setp Failed")

            print(response)
            raise KeyboardInterrupt


    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))



def run(ACCESS_URI, server_class=HTTPServer, handler_class=S, port=5001):
    open_new(ACCESS_URI)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()



def auth_call(client_id, client_secret):
    client_info = {
        'client_id': client_id,
        'client_secret':client_secret
    }
    ACCESS_URI = ('http://127.0.0.1:5000/' 
            + 'oauth/authorize?client_id=' + client_info['client_id'] + '&scope=openid+profile&response_type=code&nonce=abc')
    run(ACCESS_URI)
