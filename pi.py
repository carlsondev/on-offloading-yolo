import sys
import os

sys.path.append("/usr/local/lib/python3.6/site-packages/cv2/python-3.6")

from pi.pi_obj import Pi

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python pi.py path/to/video.mp4 [--onboard]")
        exit(1)

    args = sys.argv[1:]

    should_off = True
    if len(args) > 1 and args[1] == "--onboard":
        should_off = False

    pi_obj = Pi(args[0], should_off, "127.0.0.1", 9999)

    pi_obj.exec()
