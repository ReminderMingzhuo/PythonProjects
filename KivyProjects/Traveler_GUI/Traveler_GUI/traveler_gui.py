#!/usr/bin/env python3
import sys
sys.argv = [sys.argv[0]]

from kivy.config import Config
Config.set('graphics', 'width', '1320')
Config.set('graphics', 'height', '800')
from typing import Text
from kivy.metrics import dp
import threading
import time
from kivymd.uix import textfield
import rclpy
from rclpy import publisher
from rclpy.node import Node
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.list import OneLineListItem
from kivymd.uix.textfield import MDTextField
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from control_msgs.msg import DynamicJointState
import weakref
import kivymd_extensions.akivymd
from kivymd.uix.card import MDCard
from kivymd.uix.behaviors import RoundedRectangularElevationBehavior
from kivy.properties import StringProperty
from kivy.utils import get_color_from_hex
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.menu import MDDropdownMenu
from kivy.factory import Factory
from kivy.uix.image import Image
from kivymd.uix.list import IRightBodyTouch, ILeftBody
from kivymd.uix.selectioncontrol import MDCheckbox
from std_msgs.msg import Float64MultiArray

class Card(MDCard, RoundedRectangularElevationBehavior):
    '''Implements a material design v3 card.'''

    text = StringProperty()

class IconListItem(OneLineIconListItem):
    icon = StringProperty()

class MyCheckbox(IRightBodyTouch, MDCheckbox):
    pass


class MyAvatar(ILeftBody, Image):
    pass

class controlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/control_message', 10)
        self.linear_subscription = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.handle_joint_state,
            10)

        self.angular_subscription = self.create_subscription(
            Float32,
            '/scooby/linear_error',
            self.angular_callback,
            10)

        self.temperature = 0.0
        self.controller_error = 0.0
        self.encoder_error = 0.0
        self.motor_error = 0.0
        self.angular_error = 0.0
        self.is_new_linear_error = False

    def handle_joint_state(self, msg):
        self.temperature = msg.interface_values[0].values[4]
        self.controller_error = msg.interface_values[0].values[1]
        self.encoder_error = msg.interface_values[0].values[3]
        self.motor_error = msg.interface_values[0].values[5]
        # print(msg.interface_values[0].values[2])

    def linear_callback(self, msg):
        self.linear_error = msg.data
        self.is_new_linear_error = True

    def angular_callback(self, msg):
        self.angular_error = msg.data

    def get_error(self):
        return self.error

    def get_angular_error(self):
        return self.angular_error

    def publish_control_msg(self, msg):
        self.publisher_.publish(msg)

class GuiApp(MDApp):

    def __init__(self, node):
        super().__init__()
        self.theme_cls.theme_style = "Dark" 
        self.theme_cls.primary_palette  = "Yellow"
        
        self.screen = Builder.load_file('src/Traveler_GUI/resource/style.kv')
        menu_items = [
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "Right Triangle",
                "height": dp(56),
                "on_release": lambda x="Right Triangle": self.set_item(x),
            },
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "Triangle",
                "height": dp(56),
                "on_release": lambda x="Triangle": self.set_item(x),
            },
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "Sin",
                "height": dp(56),
                "on_release": lambda x="Sin": self.set_item(x),
            }
        ]
        self.menu = MDDropdownMenu(
            caller=self.screen.ids.drop_item,
            items=menu_items,
            position="center",
            width_mult=4,
        )

        
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Raspberry: Lowlevel Controller"))
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Computer: Highlevel Controller"))
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Computer: Decison Making Support"))
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Computer: Robot Simulation"))
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Computer: Robot Path selection"))
        self.screen.ids.scroll.add_widget(Factory.ListItemWithCheckbox(text="Computer: Robot Reconstruction"))

        self.menu.bind()
        self.control_message = Float64MultiArray()
        print('aefasdf')
        self.ros_node = node
        self.errors_writer = threading.Thread(target=self.write_errors)
        self.errors_writer.start()

        
    def set_item(self, text_item):
        self.screen.ids.drop_item.set_item(text_item)
        self.menu.dismiss()
        
    def build(self):

        
        return self.screen

    def write_errors(self):
        while(True):
            # self.ros_get_logger().info('Im inside the thread')
            self.screen.ids.temperature.value = self.ros_node.temperature
            self.screen.ids.temperature_text.text = str(self.ros_node.temperature)
            self.screen.ids.controller_error.text = str(self.ros_node.controller_error)
            self.screen.ids.controller_error.text = str(self.ros_node.encoder_error)
            self.screen.ids.controller_error.text = str(self.ros_node.motor_error)
            # self.screen.ids.error_distance.text = str(self.ros_node.get_linear_error())
            # self.screen.ids.error_angle.text = str(self.ros_node.get_angular_error())
            time.sleep(1)
            rclpy.spin_once(self.ros_node)
    def on_change_speed(self):
        self.control_message.data.append(float(self.screen.ids.drag_speed.value))
        print(self.screen.ids.drag_speed.value)
        self.publish(self.control_message)
    def on_distance_save(self):
        self.linear_msg.type = 0
        self.linear_msg.p = float(self.root.ids.distance_p.text)
        self.linear_msg.i = float(self.root.ids.distance_i.text) 
        self.linear_msg.d = float(self.root.ids.distance_d.text) 
        self.linear_msg.setpoint = float(self.root.ids.distance.text)

    def on_distance_send(self):
        self.publish(self.linear_msg)

    def on_angular_save(self):
        self.angular_msg.type = 1
        self.angular_msg.p = float(self.root.ids.angular_p.text)
        self.angular_msg.i = float(self.root.ids.angular_i.text) 
        self.angular_msg.d = float(self.root.ids.angular_d.text) 
        self.angular_msg.setpoint = float(self.root.ids.angular.text)

    def on_stop(self):
        print(self.error)

    def on_angular_send(self):
        self.publish(self.angular_msg)
        
    def stop_publishing(self):
        self.thread_event.clear()

    def start_publishing(self):
        self.thread_event.set()

    def publish(self, msg):
        self.ros_node.publish_control_msg(msg)

    def on_start(self):
        pass
            # self.root.ids.traveler_picture.add_widget(Card())
            # self.root.ids.traveler_picture.add_widget(Card())
            # self.root.ids.traveler_picture.add_widget(Card())
    

def main(args=None):
    rclpy.init(args=args)

    ros_node = controlNode()
    app = GuiApp(ros_node)
    
    app.run()
    

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    #uwbPub.destroy_node()
    rclpy.shutdown()
    ros_node.destroy_node()


if __name__ == '__main__':
    main()
