import sys
import os
sys.path.append(os.path.abspath('..'))

from consolemenu import *
from consolemenu.format import *
from consolemenu.items import *
from model.train import train
from model.preprocess import Preprocessor
from model.utilities.config import config
from model.videoCapture import Stream
from model.attendance import Attendance
from model.utilities.filechooser import init_file_selector
from model.utilities.utils import load_attendance
from shutil import copy


def input_path(default):
    Screen.println(f"    {default}")
    Screen.println("    Enter Yes -> Y to proceed with current Data......")
    Screen.println("    Enter No -> N if you want to change path to data......")
    Screen.println("    or Just press enter to Simply Exit")
    choice = input("    SELECT - ").lower()
    if choice:
        if choice == "y":
            return default
        if choice == "n":
            path = input("Enter Path to Data:.......")
            if os.path.isdir(path):
                return path
    return False


def test_camera_callback():
    camera = Stream(path=0, display=True)
    camera.capture()


def add_student_callback():
    print()
    name = input("    Enter Name: ").title()
    label = input("    Enter Roll No: ")
    files = init_file_selector()
    std_dir = os.path.join(config.TRAINING_DATA, "{}{}_{}".format(config.StudentPrefix, name, label.rjust(4, "0")))

    os.mkdir(std_dir)
    for count, source in enumerate(files):
        _, extension = os.path.splitext(source)
        copy(source, os.path.join(std_dir, f"{count}{extension}"))

    Screen.clear()
    print()
    Screen.println(f"    Student named: {name} with label: {label} successfully created.")
    Screen.println(f"    Path: {os.path.basename(std_dir)}")
    input("    Press Enter to return to Main Menu............")


def train_classifier_callback():
    path = input_path(config.TRAINING_DATA)
    if path:
        # initiate data processor
        preprocessor = Preprocessor(path=path)
        preprocessor.preprocess()

        for subject in preprocessor.data:
            data = Preprocessor.to_dataframe(subject)
            train(data=data, subject=subject.name)

        Screen.clear()
        Screen.println("")
        Screen.println(f"    Training Done.")
    input("    Press Enter to return to Main Menu............")


def take_attendance_callback():
    Screen.println("    Attendance will be taken using the data found at path:.....")
    path = input_path(os.path.join(config.BASE_DIR, "test-data"))

    Screen.clear()

    subject = input("    Enter Subject Name:.......")
    subject = "General" if subject == "" else subject
    subject = Attendance.exist(subject)

    if subject and path:
        print(subject, path)
        attendance = Attendance(subject=subject, threshold=27)
        attendance.record(path=path)
        attendance.mark()

        Screen.clear()
        Screen.println(attendance.attendance_sheet)
        Screen.println(f"    Attendance Done.")

    input("    Press Enter to return to Main Menu............")


def show_attendance_callback():
    subject = input("    Enter Subject Name:.......")
    subject = "General" if subject == "" else subject
    subject = Attendance.exist(subject)

    if subject:
        attendance, _ = load_attendance(subject=subject)
        print(attendance)

    input("    Press Enter to return to Main Menu............")


def main():
    # Change some menu formatting
    menu_format = MenuFormatBuilder().set_border_style_type(MenuBorderStyleType.HEAVY_BORDER) \
        .set_prompt("SELECT>") \
        .set_title_align('center') \
        .set_subtitle_align('center') \
        .set_left_margin(4) \
        .set_right_margin(4) \
        .show_header_bottom_border(True)

    menu = ConsoleMenu("Main Menu", "Face Recognition Based Attendance System", formatter=menu_format)
    test_camera = FunctionItem("Test Camera", test_camera_callback, should_exit=True)

    # Create a menu item that calls a function
    add_student = FunctionItem("Add Student", add_student_callback)

    # Create a menu item that calls a function
    train_classifier = FunctionItem("Train Images", train_classifier_callback)

    take_attendance = FunctionItem("Take Attendance", take_attendance_callback)

    # Create a menu item that calls a function
    show_attendance = FunctionItem("Show Attendance", show_attendance_callback)

    # Add all the items to the root menu
    menu.append_item(test_camera)
    menu.append_item(add_student)
    menu.append_item(train_classifier)
    menu.append_item(take_attendance)
    menu.append_item(show_attendance)

    # Show the menu
    menu.start()
    menu.join()


if __name__ == "__main__":
    main()
