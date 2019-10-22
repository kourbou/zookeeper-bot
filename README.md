# zookeeper-bot

![demo_gif]

Tested on Linux. You'll need **PyUserInput**, **opencv-python**, **numpy** and **FFMpeg** (for screen capturing). Assuming the window is offset by 5 pixels to the right, 200 pixels down and the window size is 550x945 you'll want to do:

```
ffmpeg -framerate 30 -video_size 550x945 -f x11grab -i :0.0+5,200 -f rawvideo -pix_fmt bgr24 pipe:1 2>/dev/null | python zookeeper-bot.py 5 200 550 945
```

[demo_gif]: https://raw.githubusercontent.com/kourbou/zookeeper-bot/master/imgs/zookeeper_demo.gif
