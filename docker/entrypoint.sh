#!/bin/bash
set -e

# Source ROS 2 Humble environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace environment if built
if [ -f "/aiformula_ws/install/setup.bash" ]; then
    source "/aiformula_ws/install/setup.bash"
fi

# Start virtual display & noVNC services if DISPLAY=:1 and not already running
export DISPLAY="${DISPLAY:-:1}"

if [ ! -e "/tmp/.X11-unix/X1" ] && [ ! -e "/tmp/.X1-lock" ]; then
    # Start Xvfb (Virtual Framebuffer)
    Xvfb :1 -screen 0 1920x1080x24+32 > /tmp/xvfb.log 2>&1 &
    sleep 1

    # Start Fluxbox Window Manager
    fluxbox > /tmp/fluxbox.log 2>&1 &
    sleep 0.5

    # Start x11vnc server
    x11vnc -display :1 -forever -shared -nopw -rfbport 5900 -quiet > /tmp/x11vnc.log 2>&1 &
    sleep 0.5

    # Start websockify (noVNC web server on port 8080)
    websockify --web /usr/share/novnc 8080 localhost:5900 > /tmp/novnc.log 2>&1 &
fi

exec "$@"
