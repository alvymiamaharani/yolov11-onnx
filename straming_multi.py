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
    def __init__(self, video_paths, paths):
        Gst.init(None)
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")  # Menggunakan satu port

        # Menambahkan beberapa video sumber ke server
        for i, path in enumerate(paths):
            if i < len(video_paths):
                # Menggunakan video dari video_paths
                factory = RTSPMediaFactory(video_paths[i])
                self.server.get_mount_points().add_factory(path, factory)
                print(
                    f"Path {path} tersedia di rtsp://127.0.0.1:8554{path} menggunakan {video_paths[i]}")

        self.server.attach(None)

    def run(self):
        loop = GObject.MainLoop()
        loop.run()


if __name__ == "__main__":
    video_paths = [
        "videos/video-test-1.mp4",  # Video sumber 1
        "videos/video-test-2.mp4",  # Video sumber 2
        "videos/video-test-3.mp4"   # Video sumber 3
    ]

    paths = ["/live1", "/live2", "/live3"]  # List path yang ingin digunakan
    server = RTSPServer(video_paths, paths)
    server.run()
