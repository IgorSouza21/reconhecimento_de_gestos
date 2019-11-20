from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import time


class CameraClick(BoxLayout):
    min = 0
    max = 255

    def __init__(self, **kwargs):
        super(CameraClick, self).__init__(**kwargs)

    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['original_camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))


class CameraApp(App):

    def build(self):
        return CameraClick()


CameraApp().run()
