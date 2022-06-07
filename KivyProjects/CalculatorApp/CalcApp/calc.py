from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.core.window import Window

# Set the app size
Window.size = (500, 700)

# Designate Our kivy design file
Builder.load_file('calc.kv')


class MyLayout(Widget):
    def clear(self):
        self.ids.calc_input.text = '0'

    # Create a button pressing function
    def button_press(self, button):
        # Create a variable that contains in the text box
        prior = self.ids.calc_input.text
        #Test error first
        if "Error" in prior:
            prior = ''
            self.ids.calc_input.text == prior
        # If 0 is sitting there
        if prior == "0":
            self.ids.calc_input.text = ""
            self.ids.calc_input.text = f'{button}'
        else:
            self.ids.calc_input.text = f'{prior}{button}'

    def math_sign(self, sign):
        prior = self.ids.calc_input.text
        # Test error first
        if "Error" in prior:
            prior = ''
            self.ids.calc_input.text == prior
        # slap a plus sign to the text box
        try:
            if sign == "x":
                self.ids.calc_input.text = f'{prior}*'

            elif sign == "=":
                answer = eval(prior)
                self.ids.calc_input.text = str(answer)
            #   +
            else:
                self.ids.calc_input.text = f'{prior}{sign}'
        except:
            self.ids.calc_input.text = "Error"
        # Addition
        '''
        if sign == "=":
            if "+" in prior:
                num_list = prior.split("+")
                answer = 0.0
                # loop thru our list
                for number in num_list:
                    answer = answer + float(number)
                self.ids.calc_input.text = str(answer)
        '''


    def dot(self):
        prior = self.ids.calc_input.text
        # Test error first
        if 'Error' in prior:
            prior = ''
            self.ids.calc_input.text == prior
        #Split out text box by +
        nums_list = prior.split("+")
        #Add a decimal to the end of the text
        if "+" in prior and "." not in nums_list[-1]:
            prior = f'{prior}.'
            self.ids.calc_input.text = prior
        elif "." in prior:
            pass
        else:
            prior = f'{prior}.'
            self.ids.calc_input.text = prior

    def remove(self):
        # Test error first
        prior = self.ids.calc_input.text
        # Test error first
        if "Error" in prior:
            prior = ''
            self.ids.calc_input.text == prior
        prior = prior[:-1]
        self.ids.calc_input.text = prior

    def pos_neg(self):
        prior = self.ids.calc_input.text
        # Test error first
        if "Error" in prior:
            prior = ''
            self.ids.calc_input.text == prior
        #Test to see if there is a -
        if "-" in prior:
            self.ids.calc_input.text = f'{prior.replace("-", "")}'
        else:
            self.ids.calc_input.text = f'-{prior}'

class CalculatorApp(App):
    def build(self):
        return MyLayout()


if __name__ == '__main__':
    CalculatorApp().run()
