"""Store detected ASV hardware metadata under a stable machine name."""

import argparse

from asv.machine import Machine, MachineCollection


def main():
    """Create an ASV machine profile containing the detected hardware."""
    parser = argparse.ArgumentParser()
    parser.add_argument("machine_name")
    args = parser.parse_args()

    machine_info = Machine.get_defaults()
    machine_info["machine"] = args.machine_name
    MachineCollection.save(args.machine_name, machine_info)

    for key in ("machine", "os", "arch", "cpu", "num_cpu", "ram"):
        print(f"{key}: {machine_info[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
