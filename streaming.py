import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

class RTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, video_path):
        super(RTSPMediaFactory, self).__init__()
        self.video_path = video_path
        self.set_shared(True)

    def do_create_element(self, url):
        pipeline = (
            f"filesrc location={self.video_path} ! decodebin ! videoconvert ! "
            f"x264enc ! rtph264pay config-interval=1 pt=96 name=pay0"
        )
        return Gst.parse_launch(pipeline)

class RTSPServer:
    def __init__(self, video_path):
        Gst.init(None)
        self.server = GstRtspServer.RTSPServer()
        factory = RTSPMediaFactory(video_path)
        self.server.get_mount_points().add_factory("/live", factory)
        self.server.attach(None)
        print("RTSP Server running at rtsp://127.0.0.1:8554/live")

    def run(self):
        loop = GObject.MainLoop()
        loop.run()

if __name__ == "__main__":
    video_path = "./video-test-2.mp4"
    server = RTSPServer(video_path)
    server.run()
