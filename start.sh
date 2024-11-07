#!/bin/bash

# 获取 app.py 的进程 ID
PID=$(ps aux | grep 'app.py' | grep -v 'grep' | awk '{print $2}')

# 如果找到进程，杀死它
if [ -n "$PID" ]; then
    echo "Stopping existing app.py process (PID: $PID)..."
    kill -9 $PID
    sleep 2  # 等待进程完全结束
else
    echo "No running app.py process found."
fi

# 重新启动 app.py 并将输出重定向到 output.log
echo "Starting app.py..."
nohup python app.py > output.log 2>&1 &

echo "app.py has been restarted."