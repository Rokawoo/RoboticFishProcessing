#!/usr/bin/env bash
# The pipeline on the robot PC in one command: sources what ROS 2, MoveIt and FAIRINO's driver
# need, then runs main.py with your options. A mode that needs MoveIt starts it with the
# FR5's driver and FAIRINO's command server (Arm/ros/fr5_bringup.launch.py) unless they run
# already, and a real run calibrates the camera first while camera.calibrated is false.
#
# What it sources, in order: ROS 2 Humble (/opt/ros/humble); every `source .../setup.bash`
# line in your ~/.bashrc (what you sourced by hand before; a script doesn't read ~/.bashrc);
# the workspace with FAIRINO's driver built: $RFP_WS, ~/rfp_ws (./setup.sh --robot) or the
# first other folder in your home that has it; the Python environment (.venv).
#
#   ./robot.sh --mode check              # the arm's state, I/O and camera; nothing moves
#   ./robot.sh --mode dry-run --step     # THE ARM MOVES, vacuum off
#   ./robot.sh --mode run                # THE ARM MOVES and picks fish
#   ./robot.sh ros                       # only MoveIt and the driver, e.g. in another terminal
#   ./robot.sh ros hardware:=mock rviz:=true

# The setup files a ~/.bashrc sources, ~ and $HOME written out
bashrc_setup_files() {
    local bashrc="${1:-$HOME/.bashrc}" file
    [ -f "$bashrc" ] || return 0
    sed -nE 's/^[[:space:]]*(source|\.)[[:space:]]+"?([^"[:space:]]*setup\.bash)"?.*$/\2/p' \
        "$bashrc" | while IFS= read -r file; do
        file="${file/#\~/$HOME}"
        file="${file//\$\{HOME\}/$HOME}"
        file="${file//\$HOME/$HOME}"
        printf '%s\n' "$file"
    done
}

# The first colcon workspace with FAIRINO's driver (fairino_hardware) built
driver_workspace() {
    local candidate
    for candidate in "${RFP_WS:-}" "$HOME/rfp_ws" "$HOME"/*/; do
        candidate="${candidate%/}"
        [ -n "$candidate" ] && [ -f "$candidate/install/setup.bash" ] || continue
        if [ -d "$candidate/install/fairino_hardware" ] ||
            [ -d "$candidate/install/share/fairino_hardware" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

main() {
    local here workspace file
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ ! -f /opt/ros/humble/setup.bash ]; then
        echo "ROS 2 Humble isn't installed: ./setup.sh --robot" >&2
        return 1
    fi
    # ROS's setup scripts read unset variables: no `set -u` here
    source /opt/ros/humble/setup.bash
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            source "$file"
        fi
    done < <(bashrc_setup_files)
    if workspace="$(driver_workspace)"; then
        source "$workspace/install/setup.bash"
    elif ! ros2 pkg prefix fairino_hardware >/dev/null 2>&1; then
        echo "No workspace with FAIRINO's driver (fairino_hardware) built in your home:" >&2
        echo "./setup.sh --robot builds it in ~/rfp_ws (or set RFP_WS to yours)" >&2
        return 1
    fi
    if [ -f "$here/.venv/bin/activate" ]; then
        source "$here/.venv/bin/activate"
    fi
    if [ "${1:-}" = "ros" ]; then
        shift
        exec ros2 launch "$here/Arm/ros/fr5_bringup.launch.py" "$@"
    fi
    exec python3 "$here/main.py" "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
