#!/usr/bin/env python3
"""
Rebuild ros_messages.json, the fields of the ROS 2 messages the arm code uses, from the
packages' own .msg, .srv and .action files. fake_ros.py builds its strict message classes
from that file, so the tests catch a misspelled or mistyped field without ROS.

Usage:
    python make_ros_messages.py /opt/ros/humble/share ~/ros2_ws/install     # on the robot PC
    python make_ros_messages.py <folder with the cloned source repositories>

Run it again when the arm code uses a message it didn't before (add it to ROOTS) or when
the ROS distribution changes. Sources of the checked-in file: the humble branches of
moveit/moveit_msgs, ros2/common_interfaces and ros2/rcl_interfaces, and fairino_msgs of
FAIR-INNOVATION/frcobot_ros2. A package that isn't found becomes a placeholder that
accepts anything (octomap_msgs, object_recognition_msgs: fields the arm code never sets).
"""

import json
import re
import sys
from pathlib import Path

ROOTS = [
    "moveit_msgs/action/MoveGroup",
    "moveit_msgs/action/ExecuteTrajectory",
    "moveit_msgs/srv/ApplyPlanningScene",
    "rcl_interfaces/srv/GetParameters",
    "sensor_msgs/msg/JointState",
    "std_msgs/msg/String",
    "fairino_msgs/msg/RobotNonrtState",
    "fairino_msgs/srv/RemoteCmdInterface",
]
PRIMITIVES = {
    "bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32",
    "uint32", "int64", "uint64", "string", "wstring",
}  # fmt: skip
FIELD = re.compile(
    r"^(?P<type>[\w/]+)(?P<array>\[[^\]]*\])?\s+(?P<name>\w+)"
    r"(\s*=\s*(?P<value>.+)|\s+(?P<default>.+))?$"
)


OUTPUT = Path(__file__).resolve().parent / "ros_messages.json"


def find(folders: list[Path], package: str, kind: str, name: str) -> Path | None:
    for folder in folders:
        for path in folder.rglob(f"{name}.{kind}"):
            if path.parent.name == kind and path.parent.parent.name == package:
                return path
    return None


def parse(text: str, package: str):
    fields, constants = [], {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = FIELD.match(line)
        if m is None:
            raise ValueError(f"can't read the line '{line}' of a {package} message")
        base = m["type"]
        if base not in PRIMITIVES:
            if base == "Header":
                base = "std_msgs/Header"
            if "/" not in base:
                base = f"{package}/{base}"
            pkg, name = base.split("/")
            base = f"{pkg}/msg/{name}"
        if m["value"] is not None and m["name"].isupper():
            constants[m["name"]] = [base, m["value"].strip()]
        else:
            fields.append([m["name"], base, m["array"] or "", m["default"] or ""])
    return fields, constants


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    folders = [Path(arg).expanduser() for arg in sys.argv[1:]]
    types: dict[str, dict] = {}
    todo = list(ROOTS)
    while todo:
        full = todo.pop()
        package, kind, name = full.split("/")
        if full in types or any(k.startswith(full + "_") for k in types):
            continue
        path = find(folders, package, kind, name)
        if path is None:
            types[full] = {"fields": None, "constants": {}}  # not found: a placeholder
            continue
        parts = path.read_text(encoding="utf-8").split("\n---")
        suffixes = {"msg": [""], "srv": ["_Request", "_Response"],
                    "action": ["_Goal", "_Result", "_Feedback"]}[kind]  # fmt: skip
        for suffix, part in zip(suffixes, parts):
            fields, constants = parse(part, package)
            types[full + suffix] = {"fields": fields, "constants": constants}
            todo += [f[1] for f in fields if f[1] not in PRIMITIVES]
    lines = [f" {json.dumps(name)}: {json.dumps(types[name])}" for name in sorted(types)]
    text = "{\n" + ",\n".join(lines) + "\n}\n"  # one type per line: readable diffs
    OUTPUT.write_text(text, encoding="utf-8")
    missing = [name for name, found in types.items() if found["fields"] is None]
    print(f"✅ {len(types) - len(missing)} message types -> {OUTPUT}")
    if missing:
        print(f"   placeholders (package not found): {', '.join(missing)}")


if __name__ == "__main__":
    main()
