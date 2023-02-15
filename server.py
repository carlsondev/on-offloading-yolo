import sys
import os

sys.path.append("/usr/local/lib/python3.6/site-packages/cv2/python-3.6")

from server.server_obj import Server


if __name__ == "__main__":
    serv_obj = Server("10.42.0.1", 9999)
    serv_obj.execute()
