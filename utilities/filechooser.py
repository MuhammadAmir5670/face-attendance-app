import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib


def on_timeout():
    Gtk.main_quit()
    return False


def FileChooser():
    dialog = Gtk.FileChooserDialog(
        title="Please choose a file", action=Gtk.FileChooserAction.OPEN)
    dialog.set_select_multiple(True)
    dialog.add_buttons(
        Gtk.STOCK_CANCEL,
        Gtk.ResponseType.CANCEL,
        Gtk.STOCK_OPEN,
        Gtk.ResponseType.OK,
    )

    add_filters(dialog)

    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        files = dialog.get_filenames()
    elif response == Gtk.ResponseType.CANCEL:
        files = []

    dialog.destroy()
    GLib.timeout_add(1, on_timeout)
    return files


def add_filters(dialog):
    filter_text = Gtk.FileFilter()
    filter_text.set_name("JPG files")
    filter_text.add_mime_type("image/jpg")
    dialog.add_filter(filter_text)

    filter_py = Gtk.FileFilter()
    filter_py.set_name("JPEG files")
    filter_py.add_mime_type("image/jpeg")
    dialog.add_filter(filter_py)

    filter_py = Gtk.FileFilter()
    filter_py.set_name("PNG files")
    filter_py.add_mime_type("image/png")
    dialog.add_filter(filter_py)

    filter_any = Gtk.FileFilter()
    filter_any.set_name("Any files")
    filter_any.add_pattern("*")
    dialog.add_filter(filter_any)


def init_file_selector():
    files = FileChooser()
    Gtk.main()
    return files


if __name__ == "__main__":
    print(init_file_selector())
