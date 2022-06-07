from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty
from kivy.uix.tabbedpanel import TabbedPanel
from kivymd.app import MDApp
from kivymd.uix.behaviors import RoundedRectangularElevationBehavior
from kivymd.uix.card import MDCard

# Designate Our kivy design file
Builder.load_file('testing.kv')

class MyLayout(Widget):
    pass


class AwesomeApp(MDApp):
    def build(self):
        return MyLayout()



if __name__ == '__main__':
    AwesomeApp().run()
