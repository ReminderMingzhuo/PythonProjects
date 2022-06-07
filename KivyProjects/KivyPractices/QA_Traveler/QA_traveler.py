from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

#Define our different screens
class FirstWindow(Screen):
    def checkbox_click(self, instance, value, topping):
        if value == True:
            self.ids.submit.background_color = (1, 1, 0, 1)
        else:
            self.ids.submit.background_color = (1, 1, 1, 1)

    def clear(self):
        self.ids.submit.background_color = (1, 1, 1, 1)

class SecondWindow(Screen):
    def checkbox_click(self, instance, value, topping):
        if value == True:
            self.ids.submit.background_color = (1, 1, 0, 1)
        else:
            self.ids.submit.background_color = (1, 1, 1, 1)

    def clear(self):
        self.ids.submit.background_color = (1, 1, 1, 1)



class ThirdWindow(Screen):
    def checkbox_click(self, instance, value, topping):
        if value == True:
            self.ids.submit.background_color = (1, 1, 0, 1)
        else:
            self.ids.submit.background_color = (1, 1, 1, 1)

    def clear(self):
        self.ids.submit.background_color = (1, 1, 1, 1)


class ForthWindow(Screen):
    def clear(self):
        self.ids.submit.background_color = (1, 1, 1, 1)

class WindowManager(ScreenManager):
    pass

# Designate Our kivy design file
kv = Builder.load_file('QA_traveler.kv')



class AwesomeApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    AwesomeApp().run()



