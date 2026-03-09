worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
workers = 1
bind = "0.0.0.0:10000"
timeout = 120