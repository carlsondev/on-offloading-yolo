import sys
from pi_obj import Pi


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python pi.py path/to/video.mp4 [--onboard]")
        exit(1)

    args = sys.argv[1:]

    should_off = True
    if len(args) > 1 and args[1] == "--onboard":
        should_off = False

    pi_obj = Pi(args[0], should_off, "10.42.0.1", 9999)

    pi_obj.exec()
