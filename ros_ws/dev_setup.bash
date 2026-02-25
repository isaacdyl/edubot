#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}"
SRC_DIR="${WS_DIR}/src"
INSTALL_SETUP="${WS_DIR}/install/setup.bash"
STAMP_FILE="${WS_DIR}/build/.last_successful_build"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:"
  echo "  source ros_ws/dev_setup.bash"
  exit 1
fi

source_underlay_if_needed() {
  if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    return 0
  fi

  for distro in jazzy humble iron rolling; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      source "/opt/ros/${distro}/setup.bash"
      return 0
    fi
  done

  echo "No ROS 2 underlay found in /opt/ros. Install ROS 2 first."
  return 1
}

needs_rebuild() {
  if [[ ! -f "${INSTALL_SETUP}" ]]; then
    return 0
  fi

  if [[ ! -f "${STAMP_FILE}" ]]; then
    return 0
  fi

  if find "${SRC_DIR}" -type f -newer "${STAMP_FILE}" | read -r _; then
    return 0
  fi

  return 1
}

source_underlay_if_needed

if needs_rebuild; then
  echo "[edubot] Running colcon build (changes detected or first-time setup)..."
  (
    cd "${WS_DIR}"
    colcon build
  )
  mkdir -p "$(dirname "${STAMP_FILE}")"
  touch "${STAMP_FILE}"
else
  echo "[edubot] Reusing existing colcon build."
fi

source "${INSTALL_SETUP}"
echo "[edubot] ROS workspace sourced."