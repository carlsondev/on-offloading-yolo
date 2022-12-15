# on-offloading-yolo

## To run on the Pi locally:

* If object detection will be done locally (no external communication), run:
```bash
$ python pi/pi.py path/to/video.mp4 --onboard
```

* If object detection will be done externally (on an already started server), run:
```bash
$ python pi/pi.py path/to/video.mp4
```

## To run the offloading server

```bash
$ python server/server.py 
```