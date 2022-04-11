import util
import os
import mlflow
import matplotlib.pyplot as plt


class ImageLogger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def parse_and_log_images(self, x, y_hat, title, step: int, subscript=None, display_count=2, names=None):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': util.tensor2im(x[i]),
                'target_face': util.tensor2im(x[i]),
                'output_face': util.tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, step=step, subscript=subscript, names=names)

    def parse_and_log_images_with_source(self, x, y, y_hat, title, step: int, subscript=None, display_count=2, names=None):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': util.tensor2im(x[i]),
                'target_face': util.tensor2im(y[i]),
                'output_face': util.tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, step=step, subscript=subscript, names=names)

    def log_images(self, name, im_data, step, subscript=None, log_latest=False, names=None):
        fig = util.vis_faces(im_data, names)
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.log_dir, name, '{}_{:04d}.jpg'.format(name, step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path)
