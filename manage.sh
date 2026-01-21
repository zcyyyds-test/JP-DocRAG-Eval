#!/bin/bash

# Configuration
APP_FILE="web_demo.py"
PORT=8501
LOG_FILE="streamlit.log"
VENV_DIR=".venv"

# Ensure we are in the project root
cd "$(dirname "$0")"

case "$1" in
    start)
        if pgrep -f "streamlit run $APP_FILE" > /dev/null; then
            echo "✅ Demo is already running."
        else
            echo "🚀 Starting JP-DocRAG Demo..."
            
            # Check Env Key (Optional warning)
            if [ -z "$GEMINI_API_KEY" ]; then
                echo "⚠️  Warning: GEMINI_API_KEY is not set in this shell."
                echo "   If you haven't set it in .env or global profile, the LLM might fail."
            fi

            # Activate Venv and Run
            source $VENV_DIR/bin/activate
            nohup streamlit run $APP_FILE --server.port $PORT > $LOG_FILE 2>&1 &
            
            echo "✅ Demo started!"
            echo "   Local: http://localhost:$PORT"
            echo "   Logs:  $LOG_FILE (View with './manage.sh logs')"
        fi
        ;;
    stop)
        if pgrep -f "streamlit run $APP_FILE" > /dev/null; then
            pkill -f "streamlit run $APP_FILE"
            echo "🛑 Demo stopped."
        else
            echo "ℹ️  Demo is not running."
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        if pgrep -f "streamlit run $APP_FILE" > /dev/null; then
            PID=$(pgrep -f "streamlit run $APP_FILE" | head -n 1)
            echo "🟢 Demo is RUNNING (PID: $PID)"
            echo "   Port: $PORT"
        else
            echo "⚪️ Demo is STOPPED"
        fi
        ;;
    logs)
        echo "📄 Showing last 20 lines of log (Ctrl+C to exit)..."
        tail -n 20 -f $LOG_FILE
        ;;
    *)
        echo "Usage: ./manage.sh {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
