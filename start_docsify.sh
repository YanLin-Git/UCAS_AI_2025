# 简易版，可使用以下命令:
# python2
# nohup python -m SimpleHTTPServer 3001 1>docsify.std 2>docsify.err &

# python3
# nohup python3 -m http.server 3001 1>docsify.std 2>docsify.err &

# 为了不使用cache缓存，强制刷新，编写脚本:
nohup python service.py 1>docsify.std 2>docsify.err &
# nohup python service.py 1>/dev/null 2>&1 &
