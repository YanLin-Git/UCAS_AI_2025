#!/usr/bin/env python
from http.server import SimpleHTTPRequestHandler, HTTPServer

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_my_headers()
        SimpleHTTPRequestHandler.end_headers(self)

    def send_my_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")


if __name__ == '__main__':
    host = ('localhost', 3001)
    server = HTTPServer(host, MyHTTPRequestHandler)
    server.serve_forever()