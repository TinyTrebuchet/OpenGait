import cv2
import numpy as np

import logging
from collections import deque

class Preprocessor:

    def __init__(self, window : int = 16, verbose : bool = False) -> None:
        # Contains max dimension (width/height) of last k images
        self.window = deque()
        self.window_size = window
        self.verbose = verbose

    # Cut-out mask & resize
    def preprocess(self, img : np.ndarray, img_size : int) -> np.ndarray | None:
        assert img.ndim == 2, f"Grayscale image required. Recieved image of shape: {img.shape}"

        if img.sum() <= 10000:
            if self.verbose:
                logging.debug(f'Image sum: {img.sum()}')
            logging.warn("Image has no data.")
            return None

        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        height = y_btm - y_top
        x_sum = img.sum(axis=0)
        x_lft = (x_sum != 0).argmax(axis=0)
        x_rht = (x_sum != 0).cumsum(axis=0).argmax(axis=0)
        width = x_rht - x_lft

        self.window.append(max(width, height))
        if len(self.window) > self.window_size:
            self.window.popleft()
        dim = max(self.window)
        half_dim = dim // 2

        x_ctr = (x_lft + x_rht) // 2
        left = x_ctr - half_dim
        right = x_ctr + half_dim
        if left <= 0 or right >= img.shape[1]:
            left += half_dim
            right += half_dim
            _ = np.zeros((img.shape[0], half_dim))
            img = np.concatenate([_, img, _], axis=1)
        y_ctr = (y_top + y_btm) // 2
        top = y_ctr - half_dim
        bottom = y_ctr + half_dim
        if top <= 0 or bottom >= img.shape[0]:
            top += half_dim
            bottom += half_dim
            _ = np.zeros((half_dim, img.shape[1]))
            img = np.concatenate([_, img, _], axis=0)
        
        img = img[top : bottom+1, left : right+1]
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        return img