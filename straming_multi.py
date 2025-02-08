from gi.repository import Gst, GstRtspServer, GObject
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')


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
    def __init__(self, video_path, paths):
        Gst.init(None)
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")  # Menggunakan satu port

        for path in paths:
            factory = RTSPMediaFactory(video_path)
            self.server.get_mount_points().add_factory(path, factory)

        self.server.attach(None)
        for path in paths:
            print(f"RTSP Server running at rtsp://127.0.0.1:8554{path}")

    def run(self):
        loop = GObject.MainLoop()
        loop.run()


if __name__ == "__main__":
    video_path = "./video-test-2.mp4"
    paths = ["/live1", "/live2", "/live3"]  # List path yang ingin digunakan
    server = RTSPServer(video_path, paths)
    server.run()
