
import io
import matplotlib.pyplot as plt
# from PySide2.QtGui import QGuiApplication, QImage
from PyQt5.QtGui import QGuiApplication, QImage


def style_sheet():
    plt.style.use({
        "axes.titlesize": 16,
        "figure.titlesize": 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        "axes.labelsize": 16,
        "figure.autolayout": True,
    })


def add_clipboard_to_figures():
    # use monkey-patching to replace the original plt.figure() function with
    # our own, which supports clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)

        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf)
                QGuiApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
                buf.close()

        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig

    plt.figure = newfig


add_clipboard_to_figures()
