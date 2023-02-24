import os
import sys
import subprocess
import signal
import shutil

from pi.pi_obj import Pi

util_tester_path: str = ""


def run_pi_test(video_path: str, do_run_onboard: bool):
    _ = input("Make sure the energy measurements have been started. Press any key to continue...")

    print(f"Starting utilization tester at path {util_tester_path}...")
    devnull = open("/dev/null", "w")
    util_proc = subprocess.Popen([util_tester_path], stdout=devnull, shell=False)

    print("Utilization tester started")

    pi_obj = Pi(video_path, not do_run_onboard, "10.42.0.1", 9999)
    print(f"Running Pi on video at path {video_path} with onboard={do_run_onboard}")

    try:
        pi_obj.exec()
    except KeyboardInterrupt:
        print("Pi execution was interrupted. Closing utilization tester...")
        util_proc.send_signal(signal.SIGINT)
        return

    print("Finished Pi execution, Closing utilization tester...")
    util_proc.send_signal(signal.SIGINT)

    move_location = input(
        "Finished! Where should the results.txt and cpu_util.ssv file be moved?: "
    )
    location_path = os.path.abspath(os.path.expanduser(move_location))
    util_file_path = os.path.join(os.path.split(util_tester_path)[0], "cpu_util.ssv")
    results_file_path = os.path.join(os.getcwd(), "results.txt")

    shutil.move(util_file_path, os.path.join(location_path, "cpu_util.ssv"))
    print(f"Successfully moved {util_file_path} to {location_path}")
    shutil.move(results_file_path, os.path.join(location_path, "results.txt"))
    print(f"Successfully moved {results_file_path} to {location_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python test_runner.py path/to/video.mp4 [--onboard]")
    args = sys.argv[1:]
    pi_server_arg = args[0]

    if len(args) < 1:
        print("Must provide a video path for Pi tests")
        print("Usage: python test_runner.py path/to/video.mp4 [--onboard]")
        exit(1)

    run_onboard = False
    if len(args) == 2 and args[1] == "--onboard":
        run_onboard = True

    run_pi_test(args[0], run_onboard)
