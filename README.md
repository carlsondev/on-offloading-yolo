# on-offloading-yolo

## To run on the Pi locally:

* If object detection will be done locally (no external communication), run:
```bash
$ python pi.py path/to/video.mp4 --onboard
```

* If object detection will be done externally (on an already started server), run:
```bash
$ python pi.py path/to/video.mp4
```

### Testing

A test script is also provided to run the Pi CPU utilization program and this code at the same time, then copy the results
to a specified folder. This script the same arguments to `pi.py`
```bash
$ python test_runner.py path/to/video.mp4 [--onboard]
```

## To run the offloading server

```bash
$ python server.py 
```