import os
import sys
import subprocess
import signal
import shutil
from threading import Thread
from datetime import datetime
import csv

from pi.pi_obj import Pi

util_tester_path: str = ""
is_jetson = False


output_file_names = {
    "cpu" : "cpu_util.ssv",
    "energy" : "", # By default energy is logged externally
    "results" : "results.txt"
}


try:
    from jtop import jtop

    is_jetson = True

    output_file_names = {
        "cpu" : "cpu-log.csv",
        "energy" : "energy-log.csv",
        "results" : "results.txt"
    }
except ImportError:
    pass


global read_jetson_metrics
read_jetson_metrics = True

def run_jetson_metrics_collection():
    with jtop() as jetson:
        with open(output_file_names["energy"], "w", newline='') as energy_csv, open(output_file_names["cpu"], "w", newline='') as cpu_csv:
            energy_writer = csv.writer(energy_csv)
            energy_writer.writerow(["collection_time", "amps", "volts", "watts"])

            cpu_writer = csv.writer(cpu_csv)
            cpu_writer.writerow(["collection_time", "cpu_util"])
            
            while jetson.ok() and read_jetson_metrics:

                collection_time : datetime = datetime.now().timestamp()
                power_dict = jetson.power["tot"]
                cpu_dict = jetson.cpu["total"]


                voltage : float = power_dict['volt'] / 1000
                current_A : float = power_dict['curr'] / 1000
                power_W : float = power_dict['power'] / 1000

                energy_writer.writerow([collection_time, current_A, voltage, power_W])

                cpu_total_util = cpu_dict['user'] + cpu_dict['nice'] + cpu_dict['system']

                cpu_writer.writerow([collection_time, cpu_total_util])




def run_pi_test(video_path: str, do_run_onboard: bool):
    if is_jetson:
        print("Collecting JTOP energy and CPU measurements...")
        daemon = Thread(target=run_jetson_metrics_collection, daemon=True, name='JTOP Energy/CPU Monitor')
        daemon.start()
    else:
        _ = input(
            "Make sure the energy measurements have been started. Press any key to continue..."
        )

        # JTOP handles CPU utilization measurements on Jetson
        print(f"Starting utilization tester at path {util_tester_path}...")
        devnull = open("/dev/null", "w")
        util_proc = subprocess.Popen([util_tester_path], stdout=devnull, shell=False)

        print("Utilization tester started")

    pi_obj = Pi(video_path, not do_run_onboard, "10.42.0.1", 9999, is_jetson)
    print(f"Running Pi on video at path {video_path} with onboard={do_run_onboard}")
    global read_jetson_metrics
    try:
        pi_obj.exec()
    except KeyboardInterrupt:
        print("Pi execution was interrupted. Closing utilization tester...")
        if is_jetson:
            read_jetson_metrics = False
            daemon.join()
        else:
            util_proc.send_signal(signal.SIGINT)
        return

    print("Finished Pi execution, Closing utilization tester...")
    if is_jetson:
        read_jetson_metrics = False
        daemon.join()
    else:
        util_proc.send_signal(signal.SIGINT)
    
    active_file_names = [k for k, v in output_file_names.items() if v != ""]
    files_list_str = ', '.join(active_file_names[:-1]) + f" and {active_file_names[-1]}"

    move_location = input(
        f"Finished! Where should the {files_list_str} files be moved?: "
    )

    location_path = os.path.abspath(os.path.expanduser(move_location))

    cpu_util_file_name = output_file_names["cpu"]
    energy_file_name = output_file_names["energy"]
    results_file_name = output_file_names["results"]

    util_file_path = os.path.join(os.path.split(util_tester_path)[0], cpu_util_file_name)

    energy_file_path = os.path.join(os.getcwd(), energy_file_name)
    results_file_path = os.path.join(os.getcwd(), results_file_name)

    shutil.move(util_file_path, os.path.join(location_path, cpu_util_file_name))
    print(f"Successfully moved {util_file_path} to {location_path}")

    if is_jetson:
        shutil.move(energy_file_path, os.path.join(location_path, energy_file_name))
        print(f"Successfully moved {energy_file_path} to {location_path}")

    shutil.move(results_file_path, os.path.join(location_path, results_file_name))
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
