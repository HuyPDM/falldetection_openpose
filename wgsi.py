from gevent.pywsgi import WSGIServer
from neucopy import app

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()