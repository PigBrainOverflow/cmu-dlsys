from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms

        # unzip file and check header
        with gzip.open(image_filename, "rb") as f:
            image_bytes = f.read()
        image_header = struct.unpack(">4i", image_bytes[:16])
        image_magic_number, image_n, row_n, col_n = image_header
        if image_magic_number != 2051:
            raise Exception("Magic Number Check Failure")
        print(f"image_header: {image_header}")

        # process content
        image_content = image_bytes[16:]
        self.X = np.frombuffer(image_content, dtype=np.uint8).astype(np.float32).reshape(image_n, row_n, col_n, 1) / 255

        # unzip file and check header
        with gzip.open(label_filename, "rb") as f:
            label_bytes = f.read()
        label_header = struct.unpack(">2i", label_bytes[:8])
        label_magic_number, label_n= label_header
        if label_magic_number != 2049:
            raise Exception("Magic Number Check Failure")
        print(f"label_header: {label_header}")

        # process content
        label_content = label_bytes[8:]
        self.y = np.frombuffer(label_content, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        sample_x = self.X[index]
        # print(sample_x.shape)
        if self.transforms:
            for transform in self.transforms:
                sample_x = transform(sample_x)
        return (sample_x, self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION