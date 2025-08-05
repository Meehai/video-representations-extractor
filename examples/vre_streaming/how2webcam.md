# VRE webcam streaming

Note: also posted as a GitHub [gist](https://gist.github.com/Meehai/ab22a452ece0cd70d2c0da683d7e0122).

We want to use a webcam or (an Android device with camera acting as a webcam) and we want to read data from it and pass to `vre_streaming`.

## USB webcam

TODO

## Android phone acting as webcam webcam

### Get the android device feed using HTTP from a browser (simplest):

On android phone:
- make sure your android phone is connected to the same wi-fi/network as your laptop/PC
- enable usb debugging on your android device (google it, it's phone-dependant)
- droidcam (non-obs) version from google play
- start it and accept all its crap requirements
- on laptop: go to the IP provided in the app i.e. `http://192.168.0.101:4747`

### Connect to the android feed using HTTP and creating a Linux webcam device (/dev/videoXXX) for capturing the feed

#### v4l2loopback module
- Install `v4l2loopback` from `this repo` or ubuntu (`v4l2loopback-dkms`).
- load it on your linux kernel: `sudo rmmod v4l2loopback` followed by `sudo modprobe v4l2loopback video_nr=2 card_label="VirtualCamAA" exclusive_caps=1`
- we put VirtualCamAA to make sure it appears on `v4l2-ctl --list-devices`.
    - Note: it does't really care about your `video_nr` option but put it regardless. Here it's on `/dev/video9`. Potential output:
- Note: sometimes `rmmod` doesn't work. Use `fuser -k /dev/video*` a bunch of times until no PID is left.
- Note: sometimes `--video_nr=0` is the only way for this to work (if `v2l2-ctl --list-devices`) complains.

```
v4l2-ctl --list-devices
ipu6 (pci:pci0000:00):
	/dev/media0

VirtualCamAA (platform:v4l2loopback-000):
	/dev/video9
```

#### droidcam-cli
- install `droidcam-cli` from [this repo](https://github.com/dev47apps/droidcam-linux-client)

- Start the linux cli tool to receive data: `sudo ./droidcam-cli -dev=/dev/video9 192.168.0.101 4747` # note the IP from your device!

### Connect to the android device webcam using USB (on Linux/Ubuntu):
#### **TODO: couldn't do it yet**

On linux (ubuntu):
- install `adb`: `sudo apt install adb`
- Test it works:
```
adb devices
List of devices attached
VED7N18531004306	device
```
- Start the linux CLI tool: `sudo ./droidcam-cli adb VED7N18531004306` # note: doesn't work... idk why

### testing that it works
Assuming the setup above worked and now you have a webcam device on your linux box (i.e. `/dev/video9)`

#### Regular video tools
- Test it with cheese: `cheese --device-/dev/video9`
- Test it with ffplay: `ffplay -f v4l2 -video_size 640x480 /dev/video9`
- Test it with vre-video-player: `vre_video_player.py /dev/video9`

#### Stream from the webcam to VRE streaming tool (rgb+semantic segmentation)

Once the `/dev/video9` device is setup, you can use the `vre-streaming` from [vre](https://github.com/Meehai/video-representations-extractor/tree/master/examples/vre_streaming) like this:

```bash
VRE_DEVICE=cuda MPL=0 ./vre_streaming.py /dev/video9 cfg_rgb_safeuav.yaml | ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x360 -framerate 30 -i -
```

Enjoy.
